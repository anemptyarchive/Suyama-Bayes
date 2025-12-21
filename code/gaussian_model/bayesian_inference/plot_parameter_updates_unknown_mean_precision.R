
# 3.3.3 1次元ガウス分布の学習と予測：平均・精度が未知の場合 --------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)
library(LaplacesDemon)

# チェック用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### ・生成分布(ガウス分布)の設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01
sqrt(1 / lambda_truth) # 標準偏差


# グラフ用のxの値を作成
x_vec <- seq(
  mu_truth - 1/sqrt(lambda_truth) * 4, 
  mu_truth + 1/sqrt(lambda_truth) * 4, 
  length.out = 201
)

# 真の分布を計算:式(2.64)
model_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = dnorm(x = x_vec, mean = mu_truth, sd = 1/sqrt(lambda_truth)) # 確率密度
)

# 真の分布を作図
ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1) + # 真の分布
  scale_color_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # 線の色:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda_truth, ")")), 
       x = "x", y = "density")


### ・データの生成 -----

# (観測)データ数を指定
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda_truth))

# 観測データを格納
data_df <- tibble::tibble(x = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_histogram(data = data_df, mapping = aes(x = x, y = ..density.., fill = "data"), 
                 bins = 30) + # 観測データ:(密度)
  geom_line(data = model_df, mapping = aes(x = x, y = dens, color = "model"), 
            size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5), linetype = c("dashed", "blank")))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gaussian Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", lambda==", lambda_truth, ", N==", N, ")")), 
       x = "x", y = "density")


### ・事前分布(ガウス・ガンマ分布)の設定 -----

# μの事前分布のパラメータを指定
m    <- 0
beta <- 1

# λの事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のμの値を作成
mu_vec <- seq(mu_truth - 40, mu_truth + 40, length.out = 201)

# グラフ用のλの値を作成
lambda_vec <- seq(0, lambda_truth * 4, length.out = 201)

# グラフ用のμとλの点を作成
mu_lambda_mat <- tidyr::expand_grid(mu = mu_vec, lambda = lambda_vec) |> # 格子点を作成
  as.matrix() # マトリクスに変換

# μとλの(同時)事前分布を計算
prior_df <- tidyr::tibble(
  mu = mu_lambda_mat[, 1], # 確率変数μ
  lambda = mu_lambda_mat[, 2], # 確率変数λ
  N_dens = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)), # μの確率密度
  Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
  density = N_dens * Gam_dens # 確率密度
)

# μとλの(同時)事前分布を作図:等高線図
ggplot() + 
  geom_contour(data = prior_df, aes(x = mu, y = lambda, z = density, color = ..level.., alpha = "prior")) + # μとλの事前分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, alpha = "data"), 
             color = "red", size = 6, shape = 4) + # 真のパラメータ
  scale_alpha_manual(values = c(param = 1, prior = 1), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # (凡例表示用の黒魔術)
  guides(alpha = guide_legend(override.aes = list(shape = c(4, NA), linetype = c("blank", "solid")))) + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = paste0("list(m==", m, ", beta==", beta, ", a==", a, ", b==", b, ")")), 
       color = "density", 
       x = expression(mu), y = expression(lambda))

# μとλの(同時)事前分布を作図:塗りつぶし等高線図
ggplot() + 
  geom_contour_filled(data = prior_df, aes(x = mu, y = lambda, z = density, fill = ..level..), 
                      alpha = 0.8, size = 0) + # μとλの事前分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, color = "param"), 
             size = 6, shape = 4) + # 真のパラメータ
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = paste0("list(m==", m, ", beta==", beta, ", a==", a, ", b==", b, ")")), 
       fill = "density", 
       x = expression(mu), y = expression(lambda))


### ・事後分布(ガウス・ガンマ分布)の計算 -----

# μの事後分布のパラメータを計算:式(3.83)
beta_hat <- N + beta
m_hat    <- (sum(x_n) + beta * m) / beta_hat

# λの事後分布のパラメータを計算:式(3.88)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) + b


# μとλの(同時)事後分布を計算
posterior_df <- tidyr::tibble(
  mu = mu_lambda_mat[, 1], # 確率変数μ
  lambda = mu_lambda_mat[, 2], # 確率変数λ
  N_dens = dnorm(x = mu, mean = m_hat, sd = 1/sqrt(beta_hat*lambda)), # μの確率密度
  Gam_dens = dgamma(x = lambda, shape = a_hat, rate = b_hat), # λの確率密度
  density = N_dens * Gam_dens # 確率密度
)

# パラメータラベル用の文字列を作成
param_text <- paste0(
  "list(hat(m)==", round(m_hat, 2), ", hat(beta)==", beta_hat, 
  ", hat(a)==", a_hat, ", hat(b)==", round(b_hat, 1), ")"
)

# μとλの(同時)事後分布を作図:等高線図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = mu, y = lambda, z = density, color = ..level.., alpha = "posterior")) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, alpha = "param"), 
             color = "red", size = 6, shape = 4) + # 真のパラメータ
  scale_alpha_manual(values = c(param = 1, posterior = 1), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # (凡例表示用の黒魔術)
  guides(alpha = guide_legend(override.aes = list(shape = c(4, NA), linetype = c("blank", "solid")))) + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = param_text), 
       color = "density", 
       x = expression(mu), y = expression(lambda))

# μとλの(同時)事後分布を作図:塗りつぶし等高線図
ggplot() + 
  geom_contour_filled(data = posterior_df, aes(x = mu, y = lambda, z = density, fill = ..level..), 
                      alpha = 0.8) + # μとλの事後分布
  geom_point(mapping = aes(x = mu_truth, y = lambda_truth, color = "param"), 
             size = 6, shape = 4) + # 真のパラメータ
  scale_color_manual(breaks = "param", values = "red", labels = "true parameter", name = "") + # (凡例表示用の黒魔術)
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = parse(text = param_text), 
       fill = "density", 
       x = expression(mu), y = expression(lambda))


### ・予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_st_hat     <- m_hat
lambda_st_hat <- beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_st_hat     <- 2 * a_hat
#mu_st_hat     <- (sum(x_n) + beta * m) / (N + beta)
#numer_lambda  <- (N + beta) * (N / 2 + a)
#denom_lambda  <- (N + 1 + beta) * ((sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b)
#lambda_st_hat <- numer_lambda / denom_lambda
#nu_st_hat     <- N + 2 * a

# 予測分布を計算:式(3.76)
predict_df <- tibble::tibble(
  x = x_vec, # 確率変数
  dens = LaplacesDemon::dst(x = x_vec, mu = mu_st_hat, sigma = 1/sqrt(lambda_st_hat), nu = nu_st_hat) # 確率密度
)

# パラメータラベル用の文字列を作成
param_text <- paste0(
  "list(N==", N, ", hat(mu)[s]==", round(mu_st_hat, 2), 
  ", hat(lambda)[s]==", round(lambda_st_hat, 5), ", hat(nu)[s]==", nu_st_hat, ")"
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
       subtitle = parse(text = param_text), 
       x = "x", y = "density")


