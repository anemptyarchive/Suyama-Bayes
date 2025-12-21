
# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合 ------------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)
library(LaplacesDemon)

# チェック用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(ガウス分布)の設定 -----

# 既知の平均パラメータを指定
mu <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01
sqrt(1 / lambda_truth) # 標準偏差


# グラフ用のxの値を作成
x_vec <- seq(
  mu - 1/sqrt(lambda_truth) * 4, 
  mu + 1/sqrt(lambda_truth) * 4, 
  length.out = 501
)

# 真の分布を計算:式(2.64)
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu, sd = 1/sqrt(lambda_truth)) # 確率密度
)

# 真の分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1) + # 真の分布
  scale_color_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # 線の色:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu, ", lambda==", lambda_truth, ")")), 
       x = "x", y = "density")


### ・データの生成 -----

# (観測)データ数を指定
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu, sd = 1/sqrt(lambda_truth))


# 観測データを格納
data_df <- tibble::tibble(x = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_histogram(data = data_df, mapping = aes(x = x, y = ..density.., fill = "data"), 
                 bins = 30) + # 観測データ(密度)
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "blank")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu, ", lambda==", lambda_truth, ", N==", N, ")")), 
       x = "x", y = "density")


### ・事前分布(ガンマ分布)の設定 -----

# λの事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のλの値を作成
lambda_vec <- seq(0, lambda_truth * 4, length.out = 501)

# λの事前分布を計算:式(2.56)
prior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  dens = dgamma(x = lambda_vec, shape = a, rate = b) # 確率密度
)

# λの事前分布を作図
ggplot() + 
  geom_vline(mapping = aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = prior_df, mapping = aes(x = lambda, y = dens, color = "prior"), 
            size = 1) + # λの事前分布
  scale_color_manual(values = c(param = "red", prior = "purple"), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b), 
       x = expression(lambda), y = "density")


### ・事後分布(ガンマ分布)の計算 -----

# λの事後分布のパラメータを計算:式(3.69)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * sum((x_n - mu)^2) + b


# λの事後分布を計算:式(2.56)
posterior_df <-tibble:: tibble(
  lambda = lambda_vec, # 確率変数
  dens = dgamma(x = lambda_vec, shape = a_hat, rate = b_hat) # 確率密度
)

# λの事後分布を作図
ggplot() + 
  geom_vline(aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = posterior_df, mapping = aes(x = lambda, y = dens, color = "posterior"), 
            size = 1) + # λの事後分布
  scale_color_manual(values = c(param = "red", posterior = "purple"), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), 
                                                  linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gamma Distribution", 
       subtitle = parse(
         text = paste0("list(N==", N, ", hat(a)==", a_hat, ", hat(b)==", round(b_hat, 1), ")")
       ), 
       x = expression(lambda), y = "density")


### ・予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.79)
mu_st         <- mu
lambda_st_hat <- a_hat / b_hat
nu_st_hat     <- 2 * a_hat
#lambda_st_hat <- (N + 2 * a) / (sum((x_n - mu)^2) + 2 * b)
#nu_st_hat     <- N + 2 * a


# 予測分布を計算:式(3.76)
predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st_hat), nu = nu_st_hat) # 確率密度
)

# 予測分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  geom_line(data = predict_df, mapping = aes(x = x, y = dens, color = "predict"), 
            size = 1) + # 予測分布
  scale_color_manual(values = c(model = "red", predict = "purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Student's t Distribution", 
       subtitle = parse(
         text = paste0(
           "list(N==", N, ", mu[s]==", round(mu_st, 2), 
           ", hat(lambda)[s]==", round(lambda_st_hat, 5), ", hat(nu)[s]==", nu_st_hat, ")"
         )
       ), 
       x = "x", y = "density")


# アニメーションによる学習推移の可視化 ----------------------------------------------------

### ・モデルの設定 -----

# 既知のパラメータを指定
mu <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01

# λの事前分布のパラメータを指定
a <- 1
b <- 1

# データ数(試行回数)を指定
N <- 100


# グラフ用のλの値を作成
lambda_vec <- seq(0, lambda_truth * 5, length.out = 501)

# グラフ用のxの値を作成
x_vec <- seq(
  mu - 1/sqrt(lambda_truth) * 4, 
  mu + 1/sqrt(lambda_truth) * 4, 
  length.out = 501
)


### ・推論処理：for関数による処理 -----

# λの事前分布(ガンマ分布)を計算:式(2.56)
anime_posterior_df <- tibble::tibble(
  lambda = lambda_vec, # 確率変数
  dens = dgamma(x = lambda_vec, shape = a, rate = b), # 確率密度
  param = paste0("N=", 0, ", a=", a, ", b=", b) |> 
    as.factor() # フレーム切替用ラベル
)

# 初期値による予測分布のパラメータを計算:式(3.79)
mu_st     <- mu
lambda_st <- a / b
nu_st     <- 2 * a

# 初期値による予測分布(スチューデントのt分布)を計算:式(3.76)
anime_predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
  param = paste0("N=", 0, ", mu_s=", mu_st, ", lambda_s=", round(lambda_st, 5), ", nu_s=", round(nu_st, 2)) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_n <- rep(NA, times = N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu, sd = 1/sqrt(lambda_truth))
  
  # λの事後分布のパラメータを更新:式(3.69)
  a <- 1 / 2 + a
  b <- 0.5 * (x_n[n] - mu)^2 + b
  
  # λの事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_df <- tibble::tibble(
    lambda = lambda_vec, # 確率変数
    dens = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    param = paste0("N=", n, ", a=", a, ", b=", round(b, 1)) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.79)
  mu_st     <- mu
  lambda_st <- a / b
  nu_st     <- 2 * a
  
  # 予測分布(スチューデントのt分布)を計算:式(3.76)
  tmp_predict_df <- tibble::tibble(
    x = x_vec, # 確率変数
    dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
    param = paste0("N=", n, ", mu_s=", mu_st, ", lambda_s=", round(lambda_st, 5), ", nu_s=", round(nu_st, 2)) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 推論結果を結合
  anime_posterior_df <- rbind(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- rbind(anime_predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### ・推論処理：tidyverseパッケージによる処理 -----

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu, sd = 1/sqrt(lambda_truth))

# 試行ごとに事後分布(ガンマ分布)を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  lambda = lambda_vec # 確率変数
) |> # 全ての組み合わせを作成
  dplyr::mutate(
    a = 0.5*n + a, 
    b = c(b, 0.5*cumsum((x_n - mu)^2) + b)[n+1]
  ) |> # 事後分布のパラメータを計算:式(3.69)
  dplyr::mutate(
    dens = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    param = paste0("N=", n, ", a=", a, ", b=", round(b, 1)) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) # 事後分布を計算:式(2.56)


# 試行ごとに予測分布(スチューデントのt分布)を計算
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  x = x_vec # 確率変数
) |> # 全ての組み合わせを作成
  dplyr::mutate(
    mu_st = mu, 
    lambda_st = c(a/b, (1:N + 2*a) / (cumsum((x_n - mu)^2) + 2*b))[n+1], 
    nu_st = n + 2*a
  ) |> # 予測分布のパラメータを計算:式(3.79)
  dplyr::mutate(
    dens = LaplacesDemon::dst(x = x_vec, mu = mu_st, sigma = 1/sqrt(lambda_st), nu = nu_st), # 確率密度
    param = paste0("N=", n, ", mu_s=", mu_st, ", lambda_s=", round(lambda_st, 5), ", nu_s=", round(nu_st, 2)) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) # 事後分布を計算:式(3.76)


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  scaled_x = c(NA, 1/(x_n - mu)^2), # 偏差の2乗の逆数に変換
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# λの事後分布のアニメーションを作図
posterior_graph <- ggplot() + 
  geom_vline(mapping = aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = anime_posterior_df, mapping = aes(x = lambda, y = dens, color = "posterior"), 
            size = 1) + # λの事後分布
  geom_point(data = anime_data_df, mapping = aes(x = scaled_x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("param", "posterior", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true parameter", "posterior", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 3), 
                                                  linetype = c("dashed", "solid", "blank"), 
                                                  shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(xlim = c(0, max(lambda_vec))) + # 軸の表示範囲
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda), y = "density")

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


# 真の分布を計算
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x, mean = mu, sd = 1/sqrt(lambda_truth)) # 確率密度
)

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
)

# 過去の観測データを複製
anime_alldata_df <- tidyr::expand_grid(
  frame = 1:N, # フレーム番号
  n = 1:N # 試行回数
) |> # 全ての組み合わせを作成
  dplyr::filter(n < frame) |> # フレーム番号以前のデータを抽出
  dplyr::mutate(
    x = x_n[n], 
    param = unique(anime_predict_df[["param"]])[frame+1]
  ) # 対応するデータとラベルを抽出

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  geom_line(data = anime_predict_df, mapping = aes(x = x, y = dens, color = "predict"), 
            size = 1) + # 予測分布
  geom_point(data = anime_alldata_df, mapping = aes(x = x, y = 0), 
             color = "pink", alpha = 0.5, size = 3) + # 過去の観測データ
  geom_point(data = anime_data_df, mapping = aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("model", "predict", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true model", "predict", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 3), 
                                                  linetype = c("dashed", "solid", "blank"), 
                                                  shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  coord_cartesian(ylim = c(0, max(model_df[["dens"]])*2)) + # 軸の表示範囲
  labs(title = "Student's t Distribution", 
       subtitle = "{current_frame}", 
       x = "x", y = "density")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N+1+10, end_pause=10, fps = 10, width = 800, height = 600)


