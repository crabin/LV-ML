from liquid_identification.data_loader import load_all_raw_data
from liquid_identification.key_frequency_analysis import (
    analyze_key_frequency_bands,
    build_key_frequency_report,
    save_key_frequency_analysis,
)
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

    key_frequency_analysis = analyze_key_frequency_bands(processed)
    print()
    print(build_key_frequency_report(key_frequency_analysis))
    analysis_dir = save_key_frequency_analysis(key_frequency_analysis)
    print(f"\n关键频段分析结果已保存: {analysis_dir}")


if __name__ == "__main__":
    main()
