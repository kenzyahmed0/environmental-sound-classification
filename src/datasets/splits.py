
def create_splits(meta):
    train_df = meta[
        meta["fold"].isin([1, 2, 3, 4, 5, 6])
    ].reset_index(drop=True)

    val_df = meta[
        meta["fold"] == 7
    ].reset_index(drop=True)

    test_df = meta[
        meta["fold"].isin([8, 9, 10])
    ].reset_index(drop=True)

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    assert len(train_df) + len(val_df) + len(test_df) == len(meta)

    assert set(train_df["fold"]).isdisjoint(set(val_df["fold"]))
    assert set(train_df["fold"]).isdisjoint(set(test_df["fold"]))
    assert set(val_df["fold"]).isdisjoint(set(test_df["fold"]))

    return train_df, val_df, test_df
