import pandas as pd
import numpy as np
import psutil
import gc
import datetime
import traceback
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import time


def log_info(message):
    """
    记录日志信息到文件
    :param message: 日志信息
    """
    with open("./taxi_data_Singapore/dbscan_kmeans_logs-20250711.txt", "a") as log_file:
        log_file.write(f"[{datetime.datetime.now()}] {message}\n")


def log_system_status(step=""):
    """
    记录系统资源使用情况
    :param step: 当前步骤描述
    """
    mem = psutil.virtual_memory()
    log_info(f"[{step}] System Status: Memory used: {mem.percent}%, Available: {mem.available / 1e9:.2f} GB")
    log_info(f"[{step}] CPU usage: {psutil.cpu_percent()}%")


def find_best_dbscan_params(features, eps_values, min_samples):
    """
    使用全量 DBSCAN 找到最佳 eps 和轮廓分数，直接保存最佳聚类中心点。
    :param features: 标准化后的特征数据
    :param eps_values: eps 的值范围
    :param min_samples: DBSCAN 的 min_samples 参数
    :return: 最佳 eps、最佳轮廓分数、最佳聚类中心点
    """
    best_eps = None
    best_score = -1
    best_centers = None
    best_n_clusters = 0

    # 打开结果文件，记录每个 eps 的结果
    with open("./taxi_data_Singapore/dbscan_results-20250711.txt", "w") as result_file:
        result_file.write("EPS\tClusters\tSilhouette_Score\tNoise_Percentage\tClustered_Percentage\tStatus\tStart_Time\tEnd_Time\n")

        for eps in eps_values:
            eps = round(eps, 3)
            labels = None
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 记录开始时间
            try:
                print(f"Running DBSCAN with eps={eps}... Start: {start_time}")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(features)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                # 统计噪点和分配到聚类组的比例
                total_samples = len(labels)
                noise_count = (labels == -1).sum()
                clustered_count = total_samples - noise_count
                noise_percentage = (noise_count / total_samples) * 100
                clustered_percentage = (clustered_count / total_samples) * 100

                if n_clusters > 6:  # 只考虑聚类数大于 5 的情况
                    try:
                        score = silhouette_score(features, labels)
                        print(f"  eps={eps}, n_clusters={n_clusters}, silhouette_score={score:.4f}")
                        print(f"  Noise: {noise_percentage:.2f}%, Clustered: {clustered_percentage:.2f}%")
                        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 记录结束时间
                        result_file.write(f"{eps}\t{n_clusters}\t{score:.4f}\t{noise_percentage:.2f}%\t{clustered_percentage:.2f}%\n")
                        result_file.write(f"Noise: {noise_percentage:.2f}%, Clustered: {clustered_percentage:.2f}%\tSuccess\t{start_time}\t{end_time}\n")
                        result_file.write(f"\n")
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_n_clusters = n_clusters
                            # 直接计算并保存最佳聚类的中心点
                            best_centers = calculate_cluster_centers(features, labels)
                    except ValueError as e:
                        print(f"  Silhouette score calculation failed for eps={eps}: {e}")
                        result_file.write(f"{eps}\t{n_clusters}\tN/A\t{noise_percentage:.2f}%\t{clustered_percentage:.2f}%\tSilhouette_Failed\t{start_time}\tN/A\n")
                else:
                    print(f"  eps={eps} produced {n_clusters} clusters (too few).")
                    result_file.write(f"{eps}\t{n_clusters}\tN/A\t{noise_percentage:.2f}%\t{clustered_percentage:.2f}%\tToo_Few_Clusters\t{start_time}\tN/A\n")

            except Exception as e:
                log_info(f"Error during EPS: {eps}")
                log_info(traceback.format_exc())
                print(f"Error logged for EPS: {eps}. Continuing with the next value.")
                result_file.write(f"{eps}\tN/A\tN/A\tN/A\tN/A\tError\t{start_time}\tN/A\n")
            finally:
                if labels is not None:
                    del labels
                gc.collect()
                log_system_status(step=f"After EPS {eps}")
                
        # 在结果文件末尾记录最佳参数
        if best_eps is not None:
            result_file.write(f"\nBest EPS: {best_eps}\n")
            result_file.write(f"Best Silhouette Score: {best_score:.4f}\n")
            result_file.write(f"Best Number of Clusters: {best_n_clusters}\n")
        else:
            result_file.write("\nNo suitable EPS was found.\n")

    return best_eps, best_score, best_centers


def calculate_cluster_centers(features, labels):
    """
    根据 DBSCAN 聚类标签计算所有聚类中心点。
    :param features: 输入特征矩阵
    :param labels: 聚类标签
    :return: 聚类中心点数组
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 排除噪声点
    centers = []

    for label in unique_labels:
        cluster_points = features[labels == label]
        center = np.mean(cluster_points, axis=0)
        centers.append(center)

    centers = np.array(centers)
    print(f"Cluster centers calculated for {len(centers)} clusters.")
    return centers


if __name__ == "__main__":
    # 数据准备
    file_path = r'./taxi_data_Singapore/3-Slidng_Windows_Data_Clustering_Clean-sta.csv'
    data = pd.read_csv(file_path)
    #data = data.head(2000)
    print("Sample number:", len(data))

    # 提取特征列
    #feature_columns = [col for col in data.columns if col.startswith("lng_grid_") or col.startswith("lat_grid_") or 
                        #col.endswith("_info")] # col.startswith("Acceleration_") or col.startswith("Curvature_") or
    # 提取带有 "-sta" 的 lat_grid_* 和 lng_grid_* 列，以及 _info 列
    feature_columns = [
        col for col in data.columns 
        if (col.startswith("lat_grid_") and col.endswith("-sta")) 
        or (col.startswith("lng_grid_") and col.endswith("-sta"))
        or col.endswith("_info")
        or col.endswith("lng_grid_0")
        or col.endswith("lat_grid_0")
        # or col.startswith("Acceleration_")
        # or col.startswith("Curvature_")
    ]
    
    print("len(feature_columns):",len(feature_columns))
    features = data[feature_columns]

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Feature scaling completed.")

    #Truncated SVD 降维
    svd = TruncatedSVD(n_components=10, random_state=2)
    scaled_features = svd.fit_transform(scaled_features)
    print(f"Reduced features to {scaled_features.shape[1]} dimensions using Truncated SVD.")
    
    # 使用全量 DBSCAN 确定最佳聚类参数并保存中心点
    #eps_values = np.arange(1.04, 1.05, 0.08)
    eps_values = np.arange(0.6, 2, 0.1)
    min_samples = 150
    print("Finding the best DBSCAN parameters...")
    best_eps, best_score, best_centers = find_best_dbscan_params(scaled_features, eps_values, min_samples=min_samples)

    if best_eps is None:
        with open("dbscan_errors_summary.txt", "w") as error_file:
            error_file.write("No suitable epsilon found.\n")
    else:
        # 保存最佳聚类中心点
        pd.DataFrame(best_centers).to_csv("./taxi_data_Singapore/Best_DBSCAN_Centers-20250711.csv", index=False, header=False)
        #print("Best DBSCAN cluster centers saved to Best_DBSCAN_Centers_2.csv.")

        # 使用 KMeans 聚类（基于 DBSCAN 中心点）
        print("Clustering with KMeans using DBSCAN centers...")
        kmeans = KMeans(n_clusters=len(best_centers), init=best_centers, n_init=1, random_state=42)
        kmeans_labels = kmeans.fit_predict(scaled_features)

        # 保存最终聚类结果
        data['kmeans_cluster'] = kmeans_labels
        df_kmeans = data[['trj_id', 'kmeans_cluster']]
        df_kmeans.to_csv("./taxi_data_Singapore/KMeans_Clustering_Results-20250711.csv", index=False)
        #print("KMeans clustering results saved to KMeans_Clustering_Results_2.csv.")
        
        
        # 确保trj_id在两个数据框中是相同的类型
        data['trj_id'] = data['trj_id'].astype(str)
        df_kmeans['trj_id'] = df_kmeans['trj_id'].astype(str)

        # 检查并删除主数据框中的kmeans_cluster列（如果存在）
        if 'kmeans_cluster' in data.columns:
            data = data.drop(columns=['kmeans_cluster'])
            print("已删除主文件中的原有 kmeans_cluster 列")

        # 合并数据，保留主文件的所有行
        merged_df = pd.merge(data, df_kmeans, on='trj_id', how='left')

        # 检查是否有未匹配的trj_id
        unmatched = data[~data['trj_id'].isin(df_kmeans['trj_id'])]
        if not unmatched.empty:
            print(f"警告：{len(unmatched)} 条记录在KMeans结果中未找到匹配的trj_id")

        # 保存合并后的文件
        output_filename = './taxi_data_Singapore/4-Slidng_Windows_Data_Clustering_Clean-sta-Cluster-Results-20250711.csv'
        merged_df.to_csv(output_filename, index=False)

        print(f"合并完成，结果已保存到 {output_filename}")
        print(f"原始主文件记录数: {len(data)}")
        print(f"合并后文件记录数: {len(merged_df)}")