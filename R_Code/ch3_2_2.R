
# 3.2.2 カテゴリ分布の学習と予測 -------------------------------------------------------------------

# 3.2.2項で利用するパッケージ
library(tidyverse)


### 尤度(カテゴリ分布)の設定 -----

# 次元数を設定:(固定)
K <- 3

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)


# 尤度のデータフレームを作成
model_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_truth_k # 確率
)

# 尤度を作図
ggplot(model_df, aes(x = k, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "blue") + # 尤度
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Catgorical Distribution", 
       subtitle = paste0("pi=(", paste0(pi_truth_k, collapse = ", "), ")"))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# カテゴリ分布に従うデータを生成
s_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()

# 観測データを確認
colSums(s_nk)


# 観測データのデータフレームを作成
data_df <- tibble(
  k = 1:K, # 次元番号
  count = colSums(s_nk) # 各次元となったデータ数
)

# 観測データのヒストグラムを作成
ggplot() + 
  #geom_bar(data = data_df, aes(x = k, y = count), 
  #         stat = "identity", position = "dodge") + # 観測データ:(度数)
  geom_bar(data = data_df, aes(x = k, y = count / N), 
           stat = "identity", position = "dodge") + # 観測データ:(相対度数)
  geom_bar(data = model_df, aes(x = k, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Catgorical Distribution", 
       subtitle = paste0("N=", N, ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"))


### 事前分布(ディリクレ分布)の設定 -----

# 事前分布のパラメータを指定
alpha_k <- c(1, 1, 1)


# 作図用のpiの値を満遍なく作成
pi_point_vec <- seq(0, 1, by = 0.02) # piがとり得る値を設定
n_point <- length(pi_point_vec) # 点の数
pi_point_df <- tibble(
  pi_1 = rep(rep(pi_point_vec, times = n_point), times = n_point), 
  pi_2 = rep(rep(pi_point_vec, each = n_point), times = n_point), 
  pi_3 = rep(pi_point_vec, each = n_point^2)
)
pi_point_df <- pi_point_df / rowSums(pi_point_df) # 正規化

# 点を間引く(ハイスぺ機なら不要…)
pi_point_mat <- pi_point_df[-1, ] %>%  # (0, 0, 0)の行を除去
  round(3) %>% # 値を丸め込み
  dplyr::distinct(pi_1, pi_2, pi_3) %>% # 重複を除去
  as.matrix() # マトリクスに変換

# 作図用のpiの値をランダムに作成
#pi_point_mat <- runif(n = 30000 * K, min = 0, max = 1) %>% # ランダムに値を生成
#  matrix(ncol = K) # マトリクスに変換
#pi_point_mat <- pi_point_mat / rowSums(pi_point_mat) # 正規化

# 真のパラメータのデータフレームを作成
parameter_df <- tibble(
  x = pi_truth_k[2] + 0.5 * pi_truth_k[3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_truth_k[3], # 三角座標への変換
)


# 事前分布を計算:式(2.48)
prior_df <- tibble(
  x = pi_point_mat[, 2] + 0.5 * pi_point_mat[, 3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_point_mat[, 3], # 三角座標への変換
  ln_C_Dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
  density = exp(ln_C_Dir) * apply(t(pi_point_mat)^(alpha_k - 1), 2, prod) # 確率密度
)

# 事前分布を作図
ggplot() + 
  geom_point(data = prior_df, aes(x = x, y = y, color = density)) + # 事前分布
  geom_point(data = parameter_df, aes(x = x, y = y), 
             color = "black", shape = 4, size = 5) + # 真の値
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # ドットのグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 0, 1)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  labs(title = "Dirichlet Distribution", 
       subtitle = paste0("alpha=(", paste(alpha_k, collapse = ", "), ")"), 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))


### 事後分布(ディリクレ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.28)
alpha_hat_k <- colSums(s_nk) + alpha_k


# 事後分布を計算:式(2.48)
posterior_df <- tibble(
  x = pi_point_mat[, 2] + 0.5 * pi_point_mat[, 3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_point_mat[, 3], # 三角座標への変換
  ln_C_Dir = lgamma(sum(alpha_hat_k)) - sum(lgamma(alpha_hat_k)), # 正規化項(対数)
  density = exp(ln_C_Dir) * apply(t(pi_point_mat)^(alpha_hat_k - 1), 2, prod) # 確率密度
)

# 事後分布を作図
ggplot() + 
  geom_point(data = posterior_df, aes(x = x, y = y, color = density)) + # 事後分布
  geom_point(data = parameter_df, aes(x = x, y = y), 
             color = "black", shape = 4, size = 5) + # 真の値
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # ドットのグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 0, 1)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  labs(title = "Dirichlet Distribution", 
       subtitle = paste0("N=", N, ", alpha_hat=(", paste(alpha_hat_k, collapse = ", "), ")"), 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))


### 予測分布(カテゴリ分布)の計算 -----

# 予測分布のパラメータを計算:式(3.31')
pi_star_hat_k <- alpha_hat_k / sum(alpha_hat_k)
pi_star_hat_k <- (colSums(s_nk) + alpha_k) / sum(colSums(s_nk) + alpha_k)


# 予測分布のデータフレームを作成
predict_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_star_hat_k # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, aes(x = k, y = prob), 
           stat = "identity", position = "dodge", fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = k, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = paste0("N=", N, ", pi_star_hat=(", paste(round(pi_star_hat_k, 2), collapse = ", "), ")"))


# ・アニメーションによる推移の確認 -----------------------------------------------------------------

# 3.2.2項で利用するパッケージ
library(tidyverse)
library(gganimate)


### モデルの設定 -----

# 次元数を設定:(固定)
K <- 3

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)

# 事前分布のパラメータを指定
alpha_k <- c(1, 1, 1)


# 作図用のpiの値を満遍なく作成
pi_point_vec <- seq(0, 1, by = 0.025) # piがとり得る値を設定
n_point <- length(pi_point_vec) # 点の数
pi_point_df <- tibble(
  pi_1 = rep(rep(pi_point_vec, times = n_point), times = n_point), 
  pi_2 = rep(rep(pi_point_vec, each = n_point), times = n_point), 
  pi_3 = rep(pi_point_vec, each = n_point^2)
)
pi_point_df <- pi_point_df / rowSums(pi_point_df) # 正規化

# 点を間引く(ハイスぺ機なら不要…)
pi_point_mat <- pi_point_df[-1, ] %>%  # (0, 0, 0)の行を除去
  round(3) %>% # 値を丸め込み
  dplyr::distinct(pi_1, pi_2, pi_3) %>% # 重複を除去
  as.matrix() # マトリクスに変換


# 事前分布(ディリクレ分布)を計算:式(2.48)
posterior_df <- tibble(
  x = pi_point_mat[, 2] + 0.5 * pi_point_mat[, 3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_point_mat[, 3], # 三角座標への変換
  ln_C_Dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
  density = exp(ln_C_Dir) * apply(t(pi_point_mat)^(alpha_k - 1), 2, prod), # 確率密度
  label = as.factor(
    paste0("N=", 0, ", alpha=(", paste0(alpha_k, collapse = ", "), ")")
  ) # フレーム切替用のラベル
)


# 初期値による予測分布のパラメーターを計算:式(3.44)
pi_star_k <- alpha_k / sum(alpha_k)


# 初期値による予測分布のデータフレームを作成
predict_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_star_k, # 確率
  label = as.factor(
    paste0(
      "N=", 0, ", pi_star=(", paste0(round(pi_star_k, 3), collapse = ", "), ")"
    )
  ) # フレーム切替用のラベル
)


### 推論処理 -----

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を作成
s_nk <- matrix(0, nrow = N, ncol = 3)

# ベイズ推論
for(n in 1:N){
  
  # カテゴリ分布に従うデータを生成
  s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = pi_truth_k) %>% 
    as.vector()
  
  # 事後分布のパラメータを更新:式(3.28)
  alpha_k <- s_nk[n, ] + alpha_k
  
  # 事後分布を計算:式(2.48)
  tmp_posterior_df <- tibble(
    x = pi_point_mat[, 2] + 0.5 * pi_point_mat[, 3], # 三角座標への変換
    y = sqrt(3) * 0.5 * pi_point_mat[, 3], # 三角座標への変換
    ln_C_Dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
    density = exp(ln_C_Dir) * apply(t(pi_point_mat)^(alpha_k - 1), 2, prod), # 確率密度
    label = as.factor(
      paste0("N=", n, ", alpha_hat=(", paste0(alpha_k, collapse = ", "), ")")
    ) # フレーム切替用のラベル
  )
  
  # 予測分布のパラメーターを更新:式(3.31)
  pi_star_k <- alpha_k / sum(alpha_k)
  
  # 予測分布のデータフレームを作成
  tmp_predict_df <- tibble(
    k = 1:K, # 作図用の値
    prob = pi_star_k, # 確率
    label = as.factor(
      paste0(
        "N=", n, ", pi_star_hat=(", paste0(round(pi_star_k, 3), collapse = ", "), ")"
      )
    ) # フレーム切替用のラベル
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
colSums(s_nk)


### 作図処理 -----

# 真のパラメータのデータフレームを作成
parameter_df <- tibble(
  x = pi_truth_k[2] + 0.5 * pi_truth_k[3], # 三角座標への変換
  y = sqrt(3) * 0.5 * pi_truth_k[3], # 三角座標への変換
)

# 観測データのデータフレームを作成
data_df <- tibble(
  x_n = c(NA, s_nk[, 2] + 0.5 * s_nk[, 3]), # 三角座標への変換
  y_n = c(NA, sqrt(3) * 0.5 * s_nk[, 3]), # 三角座標への変換
  label = unique(posterior_df[["label"]]) # フレーム切替用のラベル
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_point(data = posterior_df, aes(x = x, y = y, color = density)) + # 事後分布
  geom_point(data = parameter_df, aes(x = x, y = y), shape = 4, size = 5) + # 真の値
  geom_point(data = data_df, aes(x = x_n, y = y_n), size = 5) + # 観測データ
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # ドットのグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 0, 1)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Dirichlet Distribution", 
       subtitle = "{current_frame}", 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = (N + 1), fps = 10)


# 尤度をN+1フレーム分に複製
model_df <- tibble(
  k = rep(1:K, times = N + 1), # 次元番号
  prob = rep(pi_truth_k, times = N + 1), # 確率
  label = predict_df[["label"]] # フレーム切替用のラベル
)

# 観測データのデータフレームを作成
k_num_n <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"] # 各データの次元番号
data_df <- tibble(
  k_n = c(NA, k_num_n), 
  label = unique(predict_df[["label"]]) # フレーム切替用のラベル
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(data = predict_df, aes(x = k, y = prob), 
           stat = "identity", position = "dodge", 
           fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = k, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  geom_point(data = data_df, aes(x = k_n, y = 0), size = 5) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Categorical Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = (N + 1), fps = 10)


