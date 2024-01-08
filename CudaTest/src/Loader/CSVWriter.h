#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// CSVデータを書き出す関数
void writeCSV(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    // ファイルを開く
    std::ofstream outputFile(filename);

    // ファイルが正しく開けたか確認
    if (!outputFile.is_open()) {
        std::cerr << "ファイルを開くことができませんでした: " << filename << std::endl;
        return;
    }

    // データをCSV形式で書き込む
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outputFile << row[i];
            // 最後の要素でない場合はカンマを追加
            if (i < row.size() - 1) {
                outputFile << ",";
            }
        }
        outputFile << "\n"; // 改行
    }

    // ファイルを閉じる
    outputFile.close();
}
