
# 3.3.1 1次元ガウス分布の学習と予測：平均が未知の場合 ----------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# チェック用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(ガウス分布)の設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 既知の精度パラメータを指定
lambda <- 0.01
sqrt(1 / lambda) # 標準偏差


# グラフ用のxの値を作成
x_vec <- seq(
  mu_truth - 1/sqrt(lambda) * 4, 
  mu_truth + 1/sqrt(lambda) * 4, 
  length.out = 501
)

# 真の分布を計算:式(2.64)
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_truth, sd = 1/sqrt(lambda)) # 確率密度
)

# 真の分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1) + # 真の分布
  scale_color_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # 線の色:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda, ")")), 
       x = "x", y = "density")


### ・データの生成 -----

# (観測)データ数を指定
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda))

# 観測データを格納
data_df <- tibble::tibble(x = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_histogram(data = data_df, aes(x = x, y = ..density.., fill = "data"), 
                 bins = 30) + # 観測データ(密度)
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "blank")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda, ", N==", N, ")")), 
       x = "x", y = "density")


### ・事前分布(ガウス分布)の設定 -----

# μの事前分布の平均パラメータを指定
m <- 0

# μの事前分布の精度パラメータを指定
lambda_mu <- 0.0016
sqrt(1 / lambda_mu) # 標準偏差


# グラフ用のμの値を作成
mu_vec <- seq(
  mu_truth - 1/sqrt(lambda_mu) * 4, 
  mu_truth + 1/sqrt(lambda_mu) * 4, 
  length.out = 501
)

# μの事前分布を計算:式(2.64)
prior_df <- tibble::tibble(
  mu = mu_vec, # 確率変数
  dens = dnorm(x = mu_vec, mean = m, sd = 1/sqrt(lambda_mu)) # 確率密度
)

# μの事前分布を作図
ggplot() + 
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = prior_df, mapping = aes(x = mu, y = dens, color = "prior"), 
            size = 1) + # μの事前分布
  scale_color_manual(values = c(param = "red", prior = "purple"), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(m==", m, ", lambda[mu]==", lambda_mu, ")")), 
       x = expression(mu), y = "density")


### ・事後分布(ガウス分布)の計算 -----

# μの事後分布のパラメータを計算:式(3.53),(3.54)
lambda_mu_hat <- N * lambda + lambda_mu
m_hat         <- (lambda * sum(x_n) + lambda_mu * m) / lambda_mu_hat


# μの事後分布を計算:式(2.64)
posterior_df <- tibble::tibble(
  mu = mu_vec, # 確率変数
  dens = dnorm(x = mu_vec, mean = m_hat, sd = 1/sqrt(lambda_mu_hat)) # 確率密度
)

# μの事後分布を作図
ggplot() + 
  geom_vline(aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = posterior_df, mapping = aes(x = mu, y = dens, color = "posterior"), 
            size = 1) + # μの事後分布
  scale_color_manual(values = c(param = "red", posterior = "purple"), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "solid")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(
         text = paste0("list(N==", N, ", hat(m)==", round(m_hat, 2), ", hat(lambda)[mu]==", round(lambda_mu_hat, 5), ")")
       ), 
       x = expression(mu), y = "density")


### ・予測分布(ガウス分布)を計算 -----

# 予測分布のパラメータを計算:式(3.62')
lambda_star_hat <- lambda * lambda_mu_hat / (lambda + lambda_mu_hat)
mu_star_hat     <- m_hat
#lambda_star_hat <- (N * lambda + lambda_mu) * lambda / ((N + 1) * lambda + lambda_mu)
#mu_star_hat     <- (lambda * sum(x_n) + lambda_mu * m) / (N * lambda + lambda_mu)

# 予測分布を計算:式(2.64)
predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_star_hat, sd = 1/sqrt(lambda_star_hat)) # 確率密度
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
  labs(title = "Gaussian Distribution", 
       subtitle = parse(
         text = paste0("list(N==", N, ", hat(mu)[s]==", round(mu_star_hat, 2), ", hat(lambda)[s]==", round(lambda_star_hat, 5), ")")
       ), 
       x = "x", y = "density")


# アニメーションによる学習推移の可視化 ----------------------------------------------------

### ・モデルの設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 既知の精度パラメータを指定
lambda <- 0.01

# μの事前分布の平均パラメータを指定
m <- 0

# μの事前分布の精度パラメータを指定
lambda_mu <- 0.0016

# データ数(試行回数)を指定
N <- 100


# グラフ用のμの値を作成
mu_vec <- seq(
  mu_truth - 1/sqrt(lambda_mu) * 2, 
  mu_truth + 1/sqrt(lambda_mu) * 2, 
  length.out = 501
)

# グラフ用のxの値を作成
x_vec <- seq(
  mu_truth - 1/sqrt(lambda) * 4, 
  mu_truth + 1/sqrt(lambda) * 4, 
  length.out = 501
)


### ・推論処理：for関数による処理 -----

# μの事前分布(ガウス分布)を計算:式(2.64)
anime_posterior_df <- tibble::tibble(
  mu = mu_vec, # 確率変数
  dens = dnorm(x = mu_vec, mean = m, sd = 1/sqrt(lambda_mu)), # 確率密度
  param = paste0("N=", 0, ", m=", m, ", lambda_mu=", lambda_mu) |> 
    as.factor() # フレーム切替用ラベル
)

# 初期値による予測分布のパラメータを計算:式(3.62)
lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
mu_star     <- m

# 初期値による予測分布(ガウス分布)を計算:式(2.64)
anime_predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_star, sd = 1/sqrt(lambda_star)),  # 確率密度
  param = paste0("N=", 0, ", mu_star=", mu_star, ", lambda_star=", round(lambda_star, 5)) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_n <- rep(NA, times = N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = 1/sqrt(lambda))
  
  # μの事後分布のパラメータを更新:式(3.53),(3.54)
  lambda_mu_old <- lambda_mu
  lambda_mu <- lambda + lambda_mu
  m         <- (lambda * x_n[n] + lambda_mu_old * m) / lambda_mu
  
  # μの事後分布(ガウス分布)を計算:式(2.64)
  tmp_posterior_df <- tibble::tibble(
    mu = mu_vec, # 確率変数
    dens = dnorm(x = mu_vec, mean = m, sd = 1/sqrt(lambda_mu)), # 確率密度
    param = paste0("N=", n, ", m=", round(m, 2), ", lambda_mu=", round(lambda_mu, 5)) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.62)
  lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
  mu_star     <- m
  
  # 予測分布(ガウス分布)を計算:式(2.64)
  tmp_predict_df <- tibble::tibble(
    x = x_vec, # 確率変数
    dens = dnorm(x = x_vec, mean = mu_star, sd = 1/sqrt(lambda_star)), # 確率密度
    param = paste0("N=", n, ", mu_star=", round(mu_star, 2), ", lambda_star=", round(lambda_star, 5)) |> 
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
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda))

# 試行ごとに事後分布(ガウス分布)を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  mu = mu_vec # 確率変数
) |> # 全ての組み合わせを作成
  dplyr::mutate(
    m = c(m, (lambda*cumsum(x_n) + lambda_mu*m) / (1:N*lambda + lambda_mu))[n+1], 
    lambda_mu = n*lambda + lambda_mu
  ) |> # 事後分布のパラメータを計算:式(3.53),(3.54)
  dplyr::mutate(
    dens = dnorm(x = mu, mean = m, sd = 1/sqrt(lambda_mu)), # 確率密度
    param = paste0("N=", n, ", m=", round(m, 2), ", lambda_mu=", round(lambda_mu, 5)) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) # 事後分布を計算:式(2.64)

# 試行ごとに予測分布(ガウス分布)を計算
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  x = x_vec # 確率変数
) |> # 全ての組み合わせを作成
  dplyr::mutate(
    mu_star = c(m, (lambda*cumsum(x_n) + lambda_mu*m) / (1:N*lambda + lambda_mu))[n+1], 
    lambda_star = lambda*(n * lambda + lambda_mu) / ((n + 1)*lambda + lambda_mu)
  ) |> # 予測分布のパラメータを計算:式(3.54),(3.62')
  dplyr::mutate(
    dens = dnorm(x = x, mean = mu_star, sd = 1/sqrt(lambda_star)), # 確率密度
    param = paste0("N=", n, ", mu_star=", round(mu_star, 2), ", lambda_star=", round(lambda_star, 5)) |> 
      (\(.){factor(., levels = unique(.))})() # フレーム切替用ラベル
  ) # 事後分布を計算:式(2.64)


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# μの事後分布のアニメーションを作図
posterior_graph <- ggplot() + 
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = anime_posterior_df, mapping = aes(x = mu, y = dens, color = "posterior"), 
            size = 1) + # μの事後分布
  geom_point(data = anime_data_df, mapping = aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("param", "posterior", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true parameter", "posterior", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 3), 
                                                  linetype = c("dashed", "solid", "blank"), 
                                                  shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu), y = "density")

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N+1+10, end_pause = 10, fps = 10, width = 800, height = 600)


# 真の分布を計算
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x, mean = mu_truth, sd = 1/sqrt(lambda)) # 確率密度
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
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = "x", y = "density")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N+1+10, end_pause=10, fps = 10, width = 800, height = 600)


