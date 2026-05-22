from liquid_identification.data_loader import load_all_raw_data
from liquid_identification.dataset_split import (
    build_dataset_split_report,
    save_dataset_splits,
    split_all_feature_sets,
)
from liquid_identification.feature_extraction import (
    build_feature_extraction_report,
    extract_all_feature_sets,
    save_feature_sets,
)
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
from liquid_identification.visualization_analysis import (
    build_visualization_report,
    run_visualization_analysis,
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

    visualization_result = run_visualization_analysis(processed)
    print()
    print(build_visualization_report(visualization_result))

    feature_sets = extract_all_feature_sets(processed)
    feature_dir = save_feature_sets(feature_sets)
    print()
    print(build_feature_extraction_report(feature_sets))
    print(f"\n特征文件已保存: {feature_dir}")

    dataset_splits = split_all_feature_sets(feature_sets)
    split_dir = save_dataset_splits(dataset_splits)
    print()
    print(build_dataset_split_report(dataset_splits))
    print(f"\n数据集划分文件已保存: {split_dir}")


if __name__ == "__main__":
    main()
