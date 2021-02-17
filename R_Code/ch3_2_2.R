
# 3.2.2 カテゴリ分布の学習と予測 -------------------------------------------------------------------

# 3.2.2項で利用するパッケージ
library(tidyverse)


### 真のモデルの設定 -----

# 次元数:(固定)
K <- 3

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)

# 観測モデル(カテゴリ分布)のデータフレームを作成
model_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_truth_k # 確率
)

# 観測モデルを作図
ggplot(model_df, aes(x = k, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "purple") + # 観測モデル
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Catgorical Distribution", 
       subtitle = paste0("pi=(", paste0(pi_truth_k, collapse = ", "), ")"))


### 観測データの生成 -----

# データ数を指定
N <- 50

# (観測)データを生成
s_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()

# 観測データを確認
colSums(s_nk)

# 観測データのヒストグラムを作図
tibble(k = 1:K, count = colSums(s_nk)) %>% 
  ggplot(aes(x = k, y = count)) + 
    geom_bar(stat = "identity", position = "dodge") + # (簡易)ヒストグラム
    labs(title = "Observation Data", 
         subtitle = paste0("N=", N, ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"))


### 事前分布の設定 -----

# 事前分布のパラメータ
alpha_k <- c(1, 1, 1)

# 作図用のpiの値を満遍なく生成
point_vec <- seq(0, 1, by = 0.025) # piがとり得る値
n_point <- length(point_vec) # 点の数
pi_point <- tibble(
  pi_1 = rep(rep(point_vec, times = n_point), times = n_point), 
  pi_2 = rep(rep(point_vec, each = n_point), times = n_point), 
  pi_3 = rep(point_vec, each = n_point^2)
)
pi_point <- pi_point / rowSums(pi_point) # 正規化

# 点を間引く(ハイスぺ機なら不要…)
pi_point <- round(pi_point, 3) %>% # 値を丸め込み
  dplyr::as_tibble() %>% # データフレームに変換
  dplyr::distinct(pi_1, pi_2, pi_3) %>% # 重複を除去
  as.matrix() # マトリクスに再変換

# 作図用のpiの値をランダムに生成
#pi_point <- seq(0, 1, 0.001) %>% # piがとり得る値
#  sample(size = 90000, replace = TRUE) %>% 
#  matrix(ncol = 3)
#pi_point <- pi_point / rowSums(pi_point) # 正規化

# 事後分布(ディリクレ分布)を計算
prior_df <- tibble(
  x = pi_point[, 2] + (pi_point[, 3] / 2), # 三角座標への変換
  y = sqrt(3) * (pi_point[, 3] / 2), # 三角座標への変換
  ln_C_dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
  density = exp(ln_C_dir) * apply(t(pi_point)^(alpha_k - 1), 2, prod) # 確率密度
)

# 事前分布を作図
ggplot() + 
  geom_point(data = prior_df, aes(x = x, y = y, color = density)) + # 事前分布
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # 散布図のグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  labs(title = "Dirichlet Distribution", 
       subtitle = paste0("N=", N, ", alpha=(", paste(alpha_k, collapse = ", "), ")"), 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))


### 事後分布の計算 -----

# 事後分布のパラメータを計算
alpha_hat_k <- colSums(s_nk) + alpha_k


# 事後分布(ディリクレ分布)を計算
posterior_df <- tibble(
  x = pi_point[, 2] + (pi_point[, 3] / 2), # 三角座標への変換
  y = sqrt(3) * (pi_point[, 3] / 2), # 三角座標への変換
  ln_C_dir = lgamma(sum(alpha_hat_k)) - sum(lgamma(alpha_hat_k)), # 正規化項(対数)
  density = exp(ln_C_dir) * apply(t(pi_point)^(alpha_hat_k - 1), 2, prod) # 確率密度
)

# 真のパラメータのデータフレームを作成
parameter_df <- tibble(
  x = pi_truth_k[2] + (pi_truth_k[3] / 2), # 三角座標への変換
  y = sqrt(3) * (pi_truth_k[3] / 2), # 三角座標への変換
)

# 事後分布を作図
ggplot() + 
  geom_point(data = posterior_df, aes(x = x, y = y, color = density)) + # 事後分布
  geom_point(data = parameter_df, aes(x = x, y = y), shape = 4, size = 5) + # 真のパラメータ
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # 散布図のグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  labs(title = "Dirichlet Distribution", 
       subtitle = paste0("N=", N, ", alpha_hat=(", paste(alpha_hat_k, collapse = ", "), ")"), 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))


### 予測分布の計算 -----

# 予測分布のパラメータを計算
pi_hat_star_k <- alpha_hat_k / sum(alpha_hat_k)
pi_hat_star_k <- (colSums(s_nk) + alpha_k) / sum(colSums(s_nk) + alpha_k)

# 予測分布(カテゴリ分布)のデータフレームを作成
predict_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_hat_star_k # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, aes(x = k, y = prob), stat = "identity", position = "dodge", 
           fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = k, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Categorical Distribution", 
       subtitle = paste0("N=", N, ", pi_hat=(", paste(round(pi_hat_star_k, 2), collapse = ", "), ")"))


# ・アニメーション -----------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 次元数:(固定)
K <- 3

# 真のパラメータを指定
pi_truth_k <- c(0.3, 0.5, 0.2)

# 事前分布のパラメータを指定
alpha_k <- c(1, 1, 1)


# 作図用のpiの値を満遍なく生成
point_vec <- seq(0, 1, by = 0.025) # piがとり得る値
n_point <- length(point_vec) # 点の数
pi_point <- tibble(
  pi_1 = rep(rep(point_vec, times = n_point), times = n_point), 
  pi_2 = rep(rep(point_vec, each = n_point), times = n_point), 
  pi_3 = rep(point_vec, each = n_point^2)
)
pi_point <- pi_point / rowSums(pi_point) # 正規化

# 点を間引く(ハイスぺ機なら不要…)
pi_point <- round(pi_point, 3) %>% # 値を丸め込み
  dplyr::as_tibble() %>% # データフレームに変換
  dplyr::distinct(pi_1, pi_2, pi_3) %>% # 重複を除去
  as.matrix() # マトリクスに再変換


# 事前分布(ディリクレ分布)を計算
posterior_df <- tibble(
  x = pi_point[, 2] + (pi_point[, 3] / 2), # 三角座標への変換
  y = sqrt(3) * (pi_point[, 3] / 2), # 三角座標への変換
  ln_C_dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
  density = exp(ln_C_dir) * apply(t(pi_point)^(alpha_k - 1), 2, prod), # 確率密度
  label = as.factor(paste0("N=", 0, ", alpha=(", paste0(alpha_k, collapse = ", "), ")")) # 試行回数とパラメータのラベル
)

# 初期値による予測分布のパラメーターを計算
pi_star_k <- alpha_k / sum(alpha_k)

# 初期値による予測分布のデータフレームを作成
predict_df <- tibble(
  k = 1:K, # 次元番号
  prob = pi_star_k, # 確率
  label = as.factor(paste0("N=", 0, ", pi_star=(", paste0(pi_star_k, collapse = ", "), ")")) # 試行回数とパラメータのラベル
)


# データ数(試行回数)を指定
N <- 100

# パラメータを推定
s_nk <- matrix(0, nrow = N, ncol = 3) # 受け皿を初期化
for(n in 1:N){
  
  # カテゴリ分布に従うデータを生成
  s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = pi_truth_k) %>% 
    as.vector()
  
  # 事後分布のパラメータを更新
  alpha_k <- s_nk[n, ] + alpha_k
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    x = pi_point[, 2] + (pi_point[, 3] / 2), # 三角座標への変換
    y = sqrt(3) * (pi_point[, 3] / 2), # 三角座標への変換
    ln_C_dir = lgamma(sum(alpha_k)) - sum(lgamma(alpha_k)), # 正規化項(対数)
    density = exp(ln_C_dir) * apply(t(pi_point)^(alpha_k - 1), 2, prod), # 確率密度
    label = as.factor(paste0("N=", n, ", alpha_hat=(", paste0(alpha_k, collapse = ", "), ")")) # 試行回数とパラメータのラベル
  )
  
  # 予測分布のパラメーターを更新
  pi_star_k <- alpha_k / sum(alpha_k)
  
  # 予測分布のデータフレームを作成
  tmp_predict_df <- tibble(
    k = 1:K, # 作図用の値
    prob = pi_star_k, # 確率
    label = as.factor(paste0(
      "N=", n, ", pi_hat_star=(", paste0(round(pi_star_k, 2), collapse = ", "), ")"
    )) # 試行回数とパラメータのラベル
  )
  
  # 結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
apply(s_nk, 2, sum)


### 作図 -----

# 真のパラメータのデータフレームを作成
parameter_df <- tibble(
  x = pi_truth_k[2] + (pi_truth_k[3] / 2), # 三角座標への変換
  y = sqrt(3) * (pi_truth_k[3] / 2), # 三角座標への変換
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_point(data = posterior_df, aes(x = x, y = y, color = density)) + # 事後分布
  geom_point(data = parameter_df, aes(x = x, y = y), shape = 4, size = 5) + # 真のパラメータ
  scale_color_gradientn(colors = c("blue", "green", "yellow", "red")) + # 散布図のグラデーション
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # x軸目盛
  scale_y_continuous(breaks = c(0, 0.87), 
                     labels = c("(1, 0, 0)", "(0, 1, 0)")) + # y軸目盛
  coord_fixed(ratio = 1) + # 縦横比
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Dirichlet Distribution", 
       subtitle = "{current_frame}", 
       x = expression(paste(pi[1], ", ", pi[2], sep = "")), 
       y = expression(paste(pi[1], ", ", pi[3], sep = "")))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = (N + 1), fps = 10)


# Nフレーム分の真のモデルを格納したデータフレームを作成
label_vec <- unique(predict_df[["label"]]) # 各試行のラベルを抽出
model_df <- tibble()
for(n in 1:(N + 1)) {
  # n番目のフレーム用に作成
  tmp_df <- tibble(
    k = 1:K, 
    prob = pi_truth_k, 
    label = label_vec[n]
  )
  
  # 結果を結合
  model_df <- rbind(model_df, tmp_df)
}

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(data = predict_df, aes(x = k, y = prob), stat = "identity", position = "dodge", 
           fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = k, y = prob), stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Categorical Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = (N + 1), fps = 10)


