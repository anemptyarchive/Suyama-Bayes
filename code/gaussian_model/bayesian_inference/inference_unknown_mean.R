
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


