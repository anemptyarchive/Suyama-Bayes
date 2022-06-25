
# 3.2.1 ベルヌーイ分布の学習と予測 ----------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# チェック用
library(ggplot2)
library(magrittr)


# ベイズ推論の実装 -------------------------------------------------------------------

### ・尤度(ベルヌーイ分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 0.25


# xがとり得る値を作成
x_vec <- 0:1

# 尤度を計算:式(2.16)
model_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = c(1 - mu_truth, mu_truth) # 確率
)

# 尤度を作図
ggplot() + 
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model"), 
           stat = "identity") + # 尤度
  scale_fill_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # バーの色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("mu==", mu_truth)), 
       x = "x", y = "probability")


### ・データの生成 -----

# (観測)データ数を指定
N <- 100


# ベルヌーイ分布に従うデータを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 観測データを集計
freq_df <- tibble::tibble(
  x = x_vec, # x軸の値
  freq = c(N - sum(x_n), sum(x_n)) # 度数
)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_bar(data = freq_df, mapping = aes(x = x, y = freq/N, fill = "data"), 
           stat = "identity") + # 観測データ(相対度数)
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", N==", N, "(", paste0(freq_df[["freq"]], collapse = ", "), "))")), 
       x = "x", y = "relative frequency")


### ・事前分布(ベータ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のmuの値を作成
mu_vec <- seq(0, 1, length.out = 501)

# 事前分布を計算:式(2.41)
prior_df <- tibble::tibble(
  mu = mu_vec, # x軸の値
  density = dbeta(x = mu_vec, shape1 = a, shape2 = b) # 確率密度
)

# 事前分布を作図
ggplot() + 
  geom_line(data = prior_df, mapping = aes(x = mu, y = density, color = "prior"), 
            size = 1) + # 事前分布
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", prior = "purple"), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Beta Distribution", 
       subtitle = parse(text = paste0("list(a==", a, ", b==", b, ")")), 
       x = expression(mu), y = "density")


### ・事後分布(ベータ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.15)
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b


# 事後分布を計算:式(2.41)
posterior_df <- tibble::tibble(
  mu = mu_vec, # x軸の値
  dens = dbeta(x = mu, shape1 = a_hat, shape2 = b_hat) # 確率密度
)

# 事後分布を作図
ggplot() + 
  geom_line(data = posterior_df, mapping = aes(x = mu, y = dens, color = "posterior"), 
            size = 1) + # 事後分布
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", posterior = "purple"), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Beta Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(a)==", a_hat, ", hat(b)==", b_hat, ")")), 
       x = expression(mu), y = "density")


### ・予測分布(ベルヌーイ分布)の計算 -----

# 予測分布のパラメータを計算:式(3.19')
mu_star_hat <- a_hat / (a_hat + b_hat)
mu_star_hat <- (sum(x_n) + a) / (N + a + b)


# 予測分布を計算:式(2.16)
predict_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = c(1 - mu_star_hat, mu_star_hat) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, mapping = aes(x = x, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "predict", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, predict ="purple"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict ="purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(mu)[s]==", round(mu_star_hat, 2), ")")), 
       x = "x", y = "probability")


# アニメーションによる学習推移の可視化 -----------------------------------------------------------------

### ・モデルの設定 -----

# 真のパラメータを指定
mu_truth <- 0.25

# 事前分布のパラメータを指定
a <- 1
b <- 1

# データ数(試行回数)を指定
N <- 300


# グラフ用のmuの値を作成
mu_vec <- seq(0, 1, length.out = 501)

# グラフ用のxの値を作成
x_vec <- 0:1


### ・推論処理：for関数による処理 -----

# 事前分布(ベータ分布)を計算:式(2.41)
anime_posterior_df <- tibble::tibble(
  mu = mu_vec, # x軸の値
  dens = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
  param = paste0("N=", 0, "=(0, 0), a=", a, ", b=", b) |> 
    as.factor() # フレーム切替用ラベル
)

# 初期値による予測分布のパラメータを計算:式(3.19)
mu_star <- a / (a + b)

# 初期値による予測分布(ベルヌーイ分布)を計算:式(2.16)
anime_predict_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = c(1 - mu_star, mu_star), # 対応する確率
  param = paste0("N=", 0, "=(0, 0), mu_star=", round(mu_star, 2)) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_n <- rep(NA, times = N)

# ベイズ推論
for(n in 1:N){
  
  # ベルヌーイ分布に従うデータを生成
  x_n[n] <- rbinom(n = 1, size = 1, prob = mu_truth)
  
  # 観測データを集計
  freq_vec = c(n - sum(x_n[1:n]), sum(x_n[1:n]))
  
  # 事後分布のパラメータを更新:式(3.15)
  a <- x_n[n] + a
  b <- 1 - x_n[n] + b
  
  # 事後分布(ベータ分布)を計算:式(2.41)
  tmp_posterior_df <- tibble::tibble(
    mu = mu_vec, # x軸の値
    dens = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
    param = paste0("N=", n, "=(", paste0(freq_vec, collapse = ", "), "), a=", a, ", b=", b) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメーターを更新:式(3.19)
  mu_star <- a / (a + b)
  
  # 予測分布(ベルヌーイ分布)を計算:式(2.16)
  tmp_predict_df <- tibble::tibble(
    x = x_vec, # x軸の値
    prob = c(1 - mu_star, mu_star), # 確率
    param = paste0("N=", n, "=(", paste0(freq_vec, collapse = ", "), ")", ", mu_star=", round(mu_star, 3)) |> 
      as.factor() # フレーム切替用ラベ
  )
  
  # n回目の結果を結合
  anime_posterior_df <- rbind(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- rbind(anime_predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


### ・推論処理：tidyverseパッケージによる処理 -----

# ベルヌーイ分布に従うデータを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 試行ごとに度数を集計:(ラベル用)
freq_vec <- tibble::tibble(
  x = c(NA, x_n), # 観測データ
  n = 0:N, # 試行回数
  freq = 1 # 集計用の値
) |> # 観測データを格納
  dplyr::right_join(tidyr::expand_grid(x = x_vec, n = 0:N), by = c("x", "n")) |> # グラフ用の値に結合
  dplyr::mutate(freq = tidyr::replace_na(freq, replace = 0)) |> # 観測にない場合の欠損値を0に置換
  dplyr::arrange(n, x) |> # 集計用に昇順に並べ替え
  dplyr::group_by(x) |> # 集計用にグループ化
  dplyr::mutate(freq = cumsum(freq)) |> # 各試行までの度数を計算
  dplyr::ungroup() |> # グループ化を解除
  tidyr::pivot_wider(
    id_cols = n, 
    names_from = x, 
    names_prefix = "x", 
    values_from = freq
  ) |> # 度数列を展開
  tidyr::unite(col = "freq", dplyr::starts_with("x"), sep = ", ") |> # 度数情報をまとめて文字列化
  dplyr::pull(freq) # ベクトルとして抽出

# 試行ごとに事後分布(ベータ分布)を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  mu = mu_vec # x軸の値
) |> 
  dplyr::mutate(
    a = c(a, cumsum(x_n) + a)[n+1], 
    b = c(b, 1:N - cumsum(x_n) + b)[n+1]
  ) |> # 事後分布のパラメータを計算:式(3.15)
  dplyr::mutate(
    dens = dbeta(x = mu, shape1 = a, shape2 = b), # 確率密度
    param = paste0("N=", n, "=(", freq_vec[n+1], "), a=", a, ", b=", b) %>% 
      factor(., levels = unique(.)) # フレーム切替用ラベル
  ) # 事後分布を計算:式(2.41)

# 試行ごとに予測分布(ベルヌーイ分布)を計算
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  x = x_vec # x軸の値
) |> 
  dplyr::mutate(
    mu_star = c(a / (a + b), (cumsum(x_n) + a) / (1:N + a + b))[n+1]
  ) |> # 予測分布のパラメータを計算算:式(3.19)
  dplyr::mutate(
    prob = dbinom(x = x, size = 1, prob = mu_star), # 確率
    param = paste0("N=", n, "=(", freq_vec[n+1], "), mu_star=", round(mu_star, 3)) %>% 
      factor(., levels = unique(.)) # フレーム切替用ラベル
  ) # 予測分布を計算:式(2.16)


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# 事後分布のアニメーションを作図
posterior_graph <- ggplot() + 
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = anime_posterior_df, mapping = aes(x = mu, y = dens, color = "posterior"), 
            size = 1) + # 事後分布
  geom_point(data = anime_data_df, mapping = aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("param", "posterior", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true parameter", "posterior", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 5), linetype = c(2, 1, 0), shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Beta Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu), y = "density")

# gif画像を作成
gganimate::animate(posterior_graph, nframes = N+1, fps = 10, width = 800, height = 600)


# 尤度をフレーム分に複製
anime_model_df <- tibble::tibble(
  x = rep(x_vec, times = N+1), # x軸の値
  prob = rep(c(1 - mu_truth, mu_truth), times = N+1), # 確率
  param = anime_predict_df[["param"]] # フレーム切替用ラベル
)

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
)

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_bar(data = anime_predict_df, mapping = aes(x = x, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = anime_model_df, aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  geom_point(data = anime_data_df, aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_fill_manual(values = c(model = NA, predict = "purple", data = "pink"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict = "purple", data = "pink"), 
                     labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(fill = guide_legend(override.aes = list(fill = c(NA, "purple", NA))), 
         color = guide_legend(override.aes = list(size = c(0.5, 0.5, 5), linetype = c(2, 0, 0), shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 1)) + # 軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = "{current_frame}", 
       x = "x", y = "probability")

# gif画像を作成
gganimate::animate(predict_graph, nframes = N+1, fps = 10, width = 600, height = 600)


