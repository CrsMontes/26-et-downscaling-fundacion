"""Transfer a fitted scikit-learn forest to Google Earth Engine.

The serializer follows the R-style tree text format accepted by
``ee.Classifier.decisionTreeEnsemble``. Its structure is adapted from the
MIT-licensed ``geemap.ml.tree_to_string`` implementation, but is limited here
to regression trees because this project predicts continuous Kc.

The production smoke test must numerically compare Earth Engine predictions
against the original local scikit-learn model before a spatial product is
accepted.
"""

from __future__ import annotations

from collections.abc import Iterable

import ee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def regression_tree_to_string(
    estimator: DecisionTreeRegressor,
    feature_names: Iterable[str],
) -> str:
    """Convert one sklearn regression tree to Earth Engine R-style text."""
    feature_names = list(feature_names)
    tree = estimator.tree_

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature_idx = tree.feature
    impurities = tree.impurity
    n_samples = tree.n_node_samples
    thresholds = tree.threshold

    features = [
        feature_names[index] if index >= 0 else "leaf"
        for index in feature_idx
    ]

    raw_values = np.squeeze(tree.value)
    if raw_values.ndim != 1:
        raise ValueError(
            "Expected a single-output regression tree. "
            f"Found value array shape {raw_values.shape}."
        )

    values = np.around(raw_values, decimals=6)

    # Iterative pre-order traversal used to recover node depth and leaves.
    node_ids = np.zeros(shape=n_nodes, dtype=np.int64)
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]

    while stack:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        node_ids[node_id] = node_id

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    table = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_depth": node_depth,
            "is_leaf": is_leaves,
            "children_left": children_left,
            "children_right": children_right,
            "value": values,
            "criterion": impurities,
            "n_samples": n_samples,
            "threshold": thresholds,
            "feature_name": features,
            "sign": ["<="] * n_nodes,
        },
        dtype="object",
    )

    # Insert a duplicate row for every right branch and change its sign.
    inserts = {}
    for row in table.itertuples():
        child_right = int(row.children_right)
        if child_right > row.Index:
            ordered_row = np.array(row, dtype=object)
            ordered_row[-1] = ">"
            inserts[child_right] = ordered_row[1:]

    table_values = table.values
    for offset, key in enumerate(sorted(inserts)):
        table_values = np.insert(
            table_values,
            key + offset,
            inserts[key],
            axis=0,
        )

    ordered = pd.DataFrame(table_values, columns=table.columns)
    max_depth = int(np.max(ordered["node_depth"].astype(int)))

    tree_string = (
        f"1) root {int(n_samples[0])} 9999 9999 "
        f"({float(impurities.sum())})\n"
    )

    previous_depth = -1
    counts: list[int] = []
    count = 0

    for row in ordered.itertuples():
        depth = int(row.node_depth)
        left = int(row.children_left)
        right = int(row.children_right)

        if left != right:
            if row.Index == 0:
                count = 2
            elif previous_depth > depth:
                depths = ordered.node_depth.values[: row.Index].astype(int)
                previous_same_depth = np.where(depths == depth)[0][-1]
                count = counts[previous_same_depth] + 1
            elif previous_depth < depth:
                count = counts[row.Index - 1] * 2
            else:
                count = counts[row.Index - 1] + 1

            if depth == max_depth - 1:
                next_row = ordered.iloc[row.Index + 1]
                value = float(next_row.value)
                samples = int(next_row.n_samples)
                criterion = float(next_row.criterion)
                tail = " *\n"
            else:
                left_rows = ordered.loc[ordered.node_id == left]
                right_rows = ordered.loc[ordered.node_id == right]

                left_is_leaf = (
                    not left_rows.empty
                    and bool(left_rows.iloc[0].is_leaf)
                    and row.Index < int(left_rows.index[0])
                    and str(row.sign) == "<="
                )
                right_is_leaf = (
                    not right_rows.empty
                    and bool(right_rows.iloc[0].is_leaf)
                    and row.Index < int(right_rows.index[0])
                    and str(row.sign) == ">"
                )

                if left_is_leaf:
                    leaf_row = left_rows.iloc[0]
                    value = float(leaf_row.value)
                    samples = int(leaf_row.n_samples)
                    criterion = float(leaf_row.criterion)
                    tail = " *\n"
                elif right_is_leaf:
                    leaf_row = right_rows.iloc[0]
                    value = float(leaf_row.value)
                    samples = int(leaf_row.n_samples)
                    criterion = float(leaf_row.criterion)
                    tail = " *\n"
                else:
                    value = float(row.value)
                    samples = int(row.n_samples)
                    criterion = float(row.criterion)
                    tail = "\n"

            spacing = (depth + 1) * "  "
            feature_name = str(row.feature_name)
            threshold = float(row.threshold)
            sign = str(row.sign)

            tree_string += (
                f"{spacing}{count}) {feature_name} {sign} "
                f"{threshold:.6f} {samples} {criterion:.4f} "
                f"{value}{tail}"
            )
            previous_depth = depth

        counts.append(count)

    return tree_string


def random_forest_to_strings(
    model: RandomForestRegressor,
    feature_names: Iterable[str],
) -> list[str]:
    """Serialize every tree in a fitted RandomForestRegressor."""
    feature_names = list(feature_names)

    if not isinstance(model, RandomForestRegressor):
        raise TypeError(
            "Production model must be sklearn.ensemble.RandomForestRegressor."
        )

    if not hasattr(model, "estimators_"):
        raise ValueError("Random Forest is not fitted.")

    if int(model.n_features_in_) != len(feature_names):
        raise ValueError(
            "Feature-count mismatch between fitted model and production schema: "
            f"{model.n_features_in_} != {len(feature_names)}."
        )

    trees = [
        regression_tree_to_string(tree, feature_names)
        for tree in model.estimators_
    ]

    if len(trees) != int(model.n_estimators):
        raise RuntimeError(
            "The Earth Engine transfer did not preserve every fitted tree."
        )

    return trees


def build_ee_regressor(
    model: RandomForestRegressor,
    feature_names: Iterable[str],
) -> tuple[ee.Classifier, list[str]]:
    """Build an Earth Engine decision-tree ensemble with all RF trees."""
    trees = random_forest_to_strings(model, feature_names)
    classifier = ee.Classifier.decisionTreeEnsemble(
        [ee.String(tree) for tree in trees]
    )
    return classifier, trees
