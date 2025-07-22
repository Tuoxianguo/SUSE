import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from pybaselines import Baseline
import joblib
import warnings


warnings.filterwarnings('ignore', category=UserWarning)



def process_xrd_file(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:

        first_line = f.readline().split()
        if not first_line:
            return None, None
        half_wave_potential = float(first_line[0])


        data = np.loadtxt(f, delimiter=None)
        angles = data[:, 0]
        intensities = data[:, 1]


    smoothed = savgol_filter(intensities, window_length=11, polyorder=3)


    baseline_fitter = Baseline(x_data=angles)
    baseline = baseline_fitter.airpls(smoothed, lam=1e7)[0]
    corrected = smoothed - baseline


    features = extract_features(angles, corrected)

    return features, half_wave_potential


def extract_features(angles, intensities):

    features = {}


    features['max_intensity'] = np.max(intensities)
    features['min_intensity'] = np.min(intensities)
    features['mean_intensity'] = np.mean(intensities)
    features['median_intensity'] = np.median(intensities)
    features['std_intensity'] = np.std(intensities)
    features['skewness'] = pd.Series(intensities).skew()
    features['kurtosis'] = pd.Series(intensities).kurtosis()
    features['integral'] = np.trapz(intensities, angles)


    pt_range = (38, 42)
    pt_mask = (angles >= pt_range[0]) & (angles <= pt_range[1])
    pt_intensities = intensities[pt_mask]
    pt_angles = angles[pt_mask]

    if len(pt_intensities) > 0:

        main_peak_idx = np.argmax(pt_intensities)
        features['main_peak_position'] = pt_angles[main_peak_idx]
        features['main_peak_intensity'] = pt_intensities[main_peak_idx]


        half_max = features['main_peak_intensity'] / 2
        left_idx = np.where(pt_intensities[:main_peak_idx] <= half_max)[0]
        right_idx = np.where(pt_intensities[main_peak_idx:] <= half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            left_angle = np.interp(half_max,
                                   pt_intensities[left_idx[-1]:main_peak_idx + 1][::-1],
                                   pt_angles[left_idx[-1]:main_peak_idx + 1][::-1])
            right_angle = np.interp(half_max,
                                    pt_intensities[main_peak_idx:main_peak_idx + right_idx[0] + 1],
                                    pt_angles[main_peak_idx:main_peak_idx + right_idx[0] + 1])
            features['main_peak_fwhm'] = right_angle - left_angle
        else:
            features['main_peak_fwhm'] = np.nan


        left_half = pt_intensities[:main_peak_idx]
        right_half = pt_intensities[main_peak_idx + 1:]
        min_len = min(len(left_half), len(right_half))
        if min_len > 0:
            left_half = left_half[-min_len:]
            right_half = right_half[:min_len]
            features['peak_symmetry'] = np.mean(np.abs(left_half - right_half))
        else:
            features['peak_symmetry'] = np.nan
    else:
        features['main_peak_position'] = np.nan
        features['main_peak_intensity'] = np.nan
        features['main_peak_fwhm'] = np.nan
        features['peak_symmetry'] = np.nan


    peaks, properties = find_peaks(intensities, height=np.max(intensities) * 0.05, distance=10, width=0)

    if len(peaks) > 0:

        sorted_peaks = peaks[np.argsort(intensities[peaks])[::-1]]


        secondary_peaks = [p for p in sorted_peaks if not pt_range[0] <= angles[p] <= pt_range[1]]


        for i in range(min(3, len(secondary_peaks))):
            features[f'secondary_peak_{i + 1}_position'] = angles[secondary_peaks[i]]
            features[f'secondary_peak_{i + 1}_intensity'] = intensities[secondary_peaks[i]]

            idx_in_peaks = np.where(peaks == secondary_peaks[i])[0][0]
            features[f'secondary_peak_{i + 1}_fwhm'] = properties['widths'][idx_in_peaks]
    else:
        for i in range(3):
            features[f'secondary_peak_{i + 1}_position'] = np.nan
            features[f'secondary_peak_{i + 1}_intensity'] = np.nan
            features[f'secondary_peak_{i + 1}_fwhm'] = np.nan


    if 'main_peak_position' in features and 'secondary_peak_1_position' in features:
        features['position_ratio_1'] = features['secondary_peak_1_position'] / features['main_peak_position']
        features['position_ratio_2'] = features['secondary_peak_2_position'] / features[
            'main_peak_position'] if 'secondary_peak_2_position' in features else np.nan
    else:
        features['position_ratio_1'] = np.nan
        features['position_ratio_2'] = np.nan


    if 'main_peak_intensity' in features and 'secondary_peak_1_intensity' in features:
        features['intensity_ratio_1'] = features['secondary_peak_1_intensity'] / features['main_peak_intensity']
        features['intensity_ratio_2'] = features['secondary_peak_2_intensity'] / features[
            'main_peak_intensity'] if 'secondary_peak_2_intensity' in features else np.nan
    else:
        features['intensity_ratio_1'] = np.nan
        features['intensity_ratio_2'] = np.nan


    if len(peaks) > 0:
        peak_areas = []
        for peak in peaks:

            left_idx = max(0, peak - 5)
            right_idx = min(len(intensities) - 1, peak + 5)
            peak_areas.append(np.trapz(intensities[left_idx:right_idx], angles[left_idx:right_idx]))


        sorted_areas = np.sort(peak_areas)[::-1]
        for i in range(min(3, len(sorted_areas))):
            features[f'peak_area_{i + 1}'] = sorted_areas[i]
    else:
        for i in range(3):
            features[f'peak_area_{i + 1}'] = np.nan

    return features



def predict_half_wave_potential(model_path, data_file):
    # loading model
    model_data = joblib.load(model_path)
    model = model_data['model']
    selected_features = model_data['features']

    # 处理数据文件
    features, true_value = process_xrd_file(data_file)

    if features is None:
        raise ValueError(f"Unable to process file: {data_file}")


    features_df = pd.DataFrame([features])


    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)


    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


    if features_df.isnull().values.any():
        print("Warning")
        features_df = features_df.fillna(0)

    poly_features = poly.fit_transform(features_df)
    poly_feature_names = poly.get_feature_names_out(features_df.columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)


    all_features_df = pd.concat([features_df, poly_df], axis=1)


    available_features = set(all_features_df.columns)
    model_features = set(selected_features)
    missing_features = model_features - available_features


    for feature in missing_features:
        all_features_df[feature] = 0


    X_selected = all_features_df[selected_features]


    predicted_value = model.predict(X_selected)[0]

    return true_value, predicted_value


if __name__ == "__main__":

    model_path = "best_extratrees_model.pkl"
    data_file = ".txt"


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file does not exist: {model_path}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"The data file does not exist: {data_file}")

    # 进行预测
    try:
        true_value, predicted_value = predict_half_wave_potential(model_path, data_file)

        # 打印结果
        print("=" * 50)
        print(f"Flie: {data_file}")
        print(f"True half wave potential: {true_value:.4f}")
        print(f"Predict half wave potential: {predicted_value:.4f}")
        print(f"absolute error: {abs(true_value - predicted_value):.4f}")
        print("=" * 50)
    except Exception as e:
        print(f"An error occurred during the prediction process: {str(e)}")
        import traceback

        traceback.print_exc()
