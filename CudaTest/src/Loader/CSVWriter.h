#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// CSV�f�[�^�������o���֐�
void writeCSV(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    // �t�@�C�����J��
    std::ofstream outputFile(filename);

    // �t�@�C�����������J�������m�F
    if (!outputFile.is_open()) {
        std::cerr << "�t�@�C�����J�����Ƃ��ł��܂���ł���: " << filename << std::endl;
        return;
    }

    // �f�[�^��CSV�`���ŏ�������
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outputFile << row[i];
            // �Ō�̗v�f�łȂ��ꍇ�̓J���}��ǉ�
            if (i < row.size() - 1) {
                outputFile << ",";
            }
        }
        outputFile << "\n"; // ���s
    }

    // �t�@�C�������
    outputFile.close();
}
