
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


