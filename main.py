from liquid_identification.data_loader import load_all_raw_data
from liquid_identification.preprocessing import (
    build_preprocessing_report,
    preprocess_spectrum_data,
    save_preprocessed_data,
)


def main():
    data = load_all_raw_data()
    processed = preprocess_spectrum_data(data)
    print(build_preprocessing_report(data, processed))
    output_path = save_preprocessed_data(processed)
    print(f"\n预处理数据已保存: {output_path}")


if __name__ == "__main__":
    main()
