from liquid_identification.data_loader import load_all_raw_data


def main():
    data = load_all_raw_data()
    summary = (
        data.groupby(["range", "liquid_type"], observed=True)
        .agg(
            rows=("amplitude_dbm", "size"),
            min_mhz=("frequency_mhz", "min"),
            max_mhz=("frequency_mhz", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
