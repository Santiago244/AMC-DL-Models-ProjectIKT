%% RadioML AMC experiment visualization
% This script reads one experiment folder produced by src/train.py and
% src/evaluate.py, then creates thesis-style diagnostic plots.
%
% Default input:
%   experiments/cnn1d_quick_20260417_071055
%
% Output:
%   PNG figures saved into visualizations/current_experiment/

clear; clc; close all;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
experimentDir = fullfile(projectRoot, "experiments", "cnn1d_baseline_20260511_180658");
outputDir = fullfile(projectRoot, "visualizations", "current_experiment");

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

historyPath = fullfile(experimentDir, "history.csv");
trainReportPath = fullfile(experimentDir, "train_report.json");
testReportPath = fullfile(experimentDir, "test_report.json");
confusionPath = fullfile(experimentDir, "confusion_matrix.csv");

history = readtable(historyPath);
trainReport = readJsonFile(trainReportPath);
testReportText = fileread(testReportPath);
testReport = jsondecode(testReportText);
confusion = readmatrix(confusionPath);

classNames = string(testReport.class_names);
if iscolumn(classNames)
    classNames = classNames;
end

metrics = computeMetricsFromConfusion(confusion);
[snrValues, snrAccuracy] = parseNumericJsonObject(testReportText, "per_snr_accuracy");

fprintf("Experiment: %s\n", string(testReport.experiment_name));
fprintf("Model type: %s\n", string(testReport.model_type));
fprintf("Test samples: %d\n", testReport.test_samples);
fprintf("Overall accuracy: %.2f %%\n", 100 * testReport.overall_accuracy);
fprintf("Macro precision: %.2f %%\n", 100 * testReport.macro_precision);
fprintf("Macro recall: %.2f %%\n", 100 * testReport.macro_recall);
fprintf("Macro F1: %.2f %%\n", 100 * testReport.macro_f1);

%% 1. Training and validation learning curves
fig = figure("Color", "w", "Position", [100, 100, 1100, 480]);
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");

nexttile;
plot(history.epoch, history.train_loss, "-o", "LineWidth", 1.8, "MarkerSize", 5);
hold on;
plot(history.epoch, history.val_loss, "-s", "LineWidth", 1.8, "MarkerSize", 5);
grid on;
xlabel("Epoch");
ylabel("Cross-entropy loss");
title("Learning Curve: Loss");
legend(["Training", "Validation"], "Location", "northeast");

nexttile;
plot(history.epoch, 100 * history.train_accuracy, "-o", "LineWidth", 1.8, "MarkerSize", 5);
hold on;
plot(history.epoch, 100 * history.val_accuracy, "-s", "LineWidth", 1.8, "MarkerSize", 5);
grid on;
xlabel("Epoch");
ylabel("Accuracy (%)");
title("Learning Curve: Accuracy");
legend(["Training", "Validation"], "Location", "southeast");

exportgraphics(fig, fullfile(outputDir, "01_learning_curves.png"), "Resolution", 200);

%% 2. Summary metrics
summaryNames = categorical(["Accuracy", "Macro precision", "Macro recall", "Macro F1"]);
summaryValues = 100 * [
    testReport.overall_accuracy;
    testReport.macro_precision;
    testReport.macro_recall;
    testReport.macro_f1
];

fig = figure("Color", "w", "Position", [100, 100, 760, 460]);
bar(summaryNames, summaryValues, 0.65);
grid on;
ylabel("Score (%)");
ylim([0, max(100, ceil(max(summaryValues) / 10) * 10)]);
title("Test-Set Classification Metrics");
text(1:numel(summaryValues), summaryValues + 1, compose("%.1f%%", summaryValues), ...
    "HorizontalAlignment", "center", "FontSize", 10);
exportgraphics(fig, fullfile(outputDir, "02_summary_metrics.png"), "Resolution", 200);

%% 3. Accuracy versus SNR
fig = figure("Color", "w", "Position", [100, 100, 900, 500]);
plot(snrValues, 100 * snrAccuracy, "-o", "LineWidth", 2.0, "MarkerSize", 5);
grid on;
xlabel("SNR (dB)");
ylabel("Accuracy (%)");
title("Test Accuracy versus SNR");
xticks(snrValues);
yline(100 / numel(classNames), "--", "Chance level", "LabelHorizontalAlignment", "left");
exportgraphics(fig, fullfile(outputDir, "03_accuracy_vs_snr.png"), "Resolution", 200);

%% 4. Per-class precision, recall, F1, and accuracy
fig = figure("Color", "w", "Position", [100, 100, 1300, 620]);
tiledlayout(2, 1, "Padding", "compact", "TileSpacing", "compact");

nexttile;
bar(100 * [metrics.precision(:), metrics.recall(:), metrics.f1(:)]);
grid on;
ylabel("Score (%)");
title("Per-Class Precision, Recall, and F1");
legend(["Precision", "Recall", "F1"], "Location", "northeastoutside");
xticks(1:numel(classNames));
xticklabels(classNames);
xtickangle(45);

nexttile;
bar(100 * metrics.accuracy, 0.75);
grid on;
ylabel("Accuracy (%)");
title("Per-Class Accuracy");
xticks(1:numel(classNames));
xticklabels(classNames);
xtickangle(45);

exportgraphics(fig, fullfile(outputDir, "04_per_class_metrics.png"), "Resolution", 200);

%% 5. Raw confusion matrix
fig = figure("Color", "w", "Position", [100, 100, 980, 860]);
imagesc(confusion);
axis image;
colormap(parula);
colorbar;
title("Confusion Matrix: Raw Counts");
xlabel("Predicted class");
ylabel("True class");
xticks(1:numel(classNames));
yticks(1:numel(classNames));
xticklabels(classNames);
yticklabels(classNames);
xtickangle(45);
exportgraphics(fig, fullfile(outputDir, "05_confusion_matrix_counts.png"), "Resolution", 220);

%% 6. Row-normalized confusion matrix
rowTotals = sum(confusion, 2);
confusionNorm = confusion ./ max(rowTotals, 1);

fig = figure("Color", "w", "Position", [100, 100, 980, 860]);
imagesc(100 * confusionNorm);
axis image;
colormap(parula);
colorbar;
title("Confusion Matrix: Row-Normalized (%)");
xlabel("Predicted class");
ylabel("True class");
xticks(1:numel(classNames));
yticks(1:numel(classNames));
xticklabels(classNames);
yticklabels(classNames);
xtickangle(45);
exportgraphics(fig, fullfile(outputDir, "06_confusion_matrix_normalized.png"), "Resolution", 220);

%% 7. Prediction bias diagnostic
trueCounts = sum(confusion, 2);
predictedCounts = sum(confusion, 1)';

fig = figure("Color", "w", "Position", [100, 100, 1300, 520]);
bar([trueCounts, predictedCounts]);
grid on;
ylabel("Number of samples");
title("True Class Distribution versus Predicted Class Distribution");
legend(["True labels", "Predicted labels"], "Location", "northeastoutside");
xticks(1:numel(classNames));
xticklabels(classNames);
xtickangle(45);
exportgraphics(fig, fullfile(outputDir, "07_prediction_bias.png"), "Resolution", 200);

%% 8. SNR regime summary
snrGroups = [
    struct("name", "Low SNR (-20 to -10 dB)", "min", -20, "max", -10)
    struct("name", "Mid SNR (-8 to 6 dB)", "min", -8, "max", 6)
    struct("name", "High SNR (8 to 30 dB)", "min", 8, "max", 30)
];

groupNames = strings(numel(snrGroups), 1);
groupAcc = zeros(numel(snrGroups), 1);
for i = 1:numel(snrGroups)
    mask = snrValues >= snrGroups(i).min & snrValues <= snrGroups(i).max;
    groupNames(i) = snrGroups(i).name;
    groupAcc(i) = mean(snrAccuracy(mask));
end

fig = figure("Color", "w", "Position", [100, 100, 850, 460]);
bar(categorical(groupNames), 100 * groupAcc, 0.6);
grid on;
ylabel("Mean accuracy across SNR values (%)");
title("Accuracy by SNR Regime");
text(1:numel(groupAcc), 100 * groupAcc + 1, compose("%.1f%%", 100 * groupAcc), ...
    "HorizontalAlignment", "center", "FontSize", 10);
exportgraphics(fig, fullfile(outputDir, "08_snr_regime_accuracy.png"), "Resolution", 200);

fprintf("Saved figures to:\n%s\n", outputDir);

%% Local helper functions
function data = readJsonFile(path)
    text = fileread(path);
    data = jsondecode(text);
end

function metrics = computeMetricsFromConfusion(confusion)
    tp = diag(confusion);
    predictedTotal = sum(confusion, 1)';
    actualTotal = sum(confusion, 2);

    precision = safeDivide(tp, predictedTotal);
    recall = safeDivide(tp, actualTotal);
    f1 = safeDivide(2 * precision .* recall, precision + recall);
    accuracy = recall;

    metrics = struct();
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.f1 = f1;
    metrics.accuracy = accuracy;
end

function out = safeDivide(num, den)
    out = zeros(size(num));
    mask = den > 0;
    out(mask) = num(mask) ./ den(mask);
end

function [keys, values] = parseNumericJsonObject(jsonText, objectName)
    pattern = """" + objectName + """\s*:\s*\{(?<body>.*?)\}";
    match = regexp(jsonText, pattern, "names", "once");
    if isempty(match)
        error("Could not find JSON object named %s.", objectName);
    end

    pairPattern = """(?<key>-?\d+)""\s*:\s*(?<value>[-+0-9.eE]+)";
    pairs = regexp(match.body, pairPattern, "names");
    keys = zeros(numel(pairs), 1);
    values = zeros(numel(pairs), 1);

    for i = 1:numel(pairs)
        keys(i) = str2double(pairs(i).key);
        values(i) = str2double(pairs(i).value);
    end

    [keys, order] = sort(keys);
    values = values(order);
end

