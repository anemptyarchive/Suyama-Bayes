
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(LaplacesDemon)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 ----------------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# 真のパラメータを指定
mu_truth     <- 5
lambda_truth <- 0.25
1/sqrt(lambda_truth) # 標準偏差


#### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_size <- (1/sqrt(lambda_truth)) |> #標準偏差
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., abs(x_n-mu_truth))})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu_truth - x_size
x_max <- mu_truth + x_size

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, length.out = 1001)


#### 分布の計算 -----

# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu_truth,     # 平均パラメータ
  lambda = lambda_truth, # 精度パラメータ
  sigma  = 1/sqrt(lambda), # 標準偏差パラメータ
  dens   = dnorm(x = x, mean = mu, sd = sigma) # 確率密度
)


### 事前分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# μの事前分布のパラメータを指定
m    <- 0
beta <- 1

# λの事前分布のパラメータを指定
a <- 1
b <- 1


#### 変数の設定 -----

# μ軸の範囲を設定
mu_min = x_min
mu_max = x_max

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 101)


# λ軸の範囲を設定
k <- 3
u <- 0.5
lambda_min <- 0
lambda_max <- lambda_truth |> # 真のパラメータ
  (\(.) {. * k})() |> # 定数倍
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 101)


#### 分布の計算 -----

# 事前分布の確率密度を計算
prior_df <- tidyr::expand_grid(
  mu     = mu_vec,    # μの確率変数
  lambda = lambda_vec # λの確率変数
) |> # 格子点を作成
  dplyr::mutate(
    m         = m,                 # 平均パラメータ
    beta      = beta,              # 係数パラメータ
    lambda_mu = beta * lambda,     # 精度パラメータ
    sigma_mu  = 1/sqrt(lambda_mu), # 標準偏差パラメータ
    a         = a,                 # 形状パラメータ
    b         = b,                 # 尺度パラメータ
    N_dens    = dnorm(x = mu, mean = m, sd = sigma_mu),  # μの確率密度
    Gam_dens  = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
    dens      = N_dens * Gam_dens # μ, λの確率密度
  )


### 事前分布(ガウス分布)の設定 -----

#### パラメータの設定 -----

# データ数を指定
N <- 50

# 観測データを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda_truth))


#### データの集計 -----

# 階級数を指定
bin_num <- 40

# 階級幅を計算
bin_size <- (x_max - x_min) / bin_num

# 境界値の範囲を設定
bin_min <- x_min - 0.5*bin_size
bin_max <- x_max + 0.5*bin_size

# 観測データを集計
obs_df <- tibble::tibble(
  x = x_n # サンプル値
) |> 
  dplyr::mutate(
    bin_i  = (x - bin_min) %/% bin_size,        # 階級番号
    center = bin_min + (bin_i + 0.5) * bin_size # 階級値
  ) |> 
  dplyr::count(
    center, name = "freq" # 度数
  ) |> 
  dplyr::mutate(
    dens = freq / (bin_size * N) # 密度
  ) |> 
  tidyr::complete(
    center = seq(from = x_min, to = x_max, by = bin_size), 
    fill = list(freq = 0, dens = 0)
  ) # 未観測値を補完


### 事後分布(ガウス・ガンマ分布)の計算 -----

#### パラメータの計算 -----

# μの事後分布のパラメータを計算:式(3.83)
beta_hat <- N + beta
m_hat    <- (sum(x_n) + beta * m) / beta_hat

# λの事後分布のパラメータを計算:式(3.88)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) + b


#### 分布の計算  -----

# 事後分布の確率密度を計算
posterior_df <- tidyr::expand_grid(
  mu     = mu_vec,    # μの確率変数
  lambda = lambda_vec # λの確率変数
) |> # 格子点を作成
  dplyr::mutate(
    m         = m_hat,             # 平均パラメータ
    beta      = beta_hat,          # 係数パラメータ
    lambda_mu = beta * lambda,     # 精度パラメータ
    sigma_mu  = 1/sqrt(lambda_mu), # 標準偏差パラメータ
    a         = a_hat,             # 形状パラメータ
    b         = b_hat,             # 尺度パラメータ
    N_dens    = dnorm(x = mu, mean = m, sd = sigma_mu),  # μの確率密度
    Gam_dens  = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
    dens      = N_dens * Gam_dens # μ, λの確率密度
  )


#### 分布の作図 -----

# 事後分布のラベルを作成
posterior_param_lbl <- paste0(
  "list(", 
  "N == ",             N, ", ", 
  "mu[truth] == ",     round(mu_truth,     digits = 2), ", ", 
  "lambda[truth] == ", round(lambda_truth, digits = 5), ", ", 
  "hat(m) == ",        round(m_hat,        digits = 2), ", ", 
  "hat(beta) == ",     round(beta_hat,     digits = 1),  ", ", 
  "hat(a) == ",        round(a_hat,        digits = 1), ", ", 
  "hat(b) == ",        round(b_hat,        digits = 1), 
  ")"
) |> 
  parse(text = _)

# 確率密度軸の範囲を設定
u <- 0.5
dens_max <- max(
  prior_df |> 
    dplyr::pull(dens) |> 
    max(), 
  posterior_df |> 
    dplyr::pull(dens) |> 
    max()
) |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# 等高線を設定
level_num     <- 11
dens_vals <- seq(from = 0, to = dens_max, length.out = level_num)
dens_vals

# 事後分布を作図
ggplot() + 
  geom_contour(
    data    = prior_df, 
    mapping = aes(x = mu, y = lambda, z = dens, color = after_stat(level), linetype = "prior"), 
    #breaks = dens_vals, # (軸の共通化用)
    bins = level_num
  ) + # 事前分布
  geom_contour_filled(
    data    = posterior_df, 
    mapping = aes(x = mu, y = lambda, z = dens, fill = after_stat(level), linetype = "posterior"), 
    breaks = dens_vals, # (軸の共通化用)
    alpha = 0.5
  ) + # 事後分布
  geom_vline(
    mapping = aes(xintercept = mu_truth, linetype = "model"), 
    color = "red", linewidth = 1
  ) + # 真のパラメータ
  geom_hline(
    mapping = aes(yintercept = lambda_truth, linetype = "model"), 
    color = "red", linewidth = 1
  ) + # 真のパラメータ
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = mu_truth, 
      labels    = expression(mu[truth])
    ) # パラメータラベル
  ) + 
  scale_y_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = lambda_truth, 
      labels    = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_viridis_c(
    limits = c(0, dens_max) # (軸の共通化用)
  ) + 
  scale_linetype_manual(
    breaks = c("model", "prior", "posterior"), 
    values = c("dashed", "dotted", "solid"), 
    labels = c("true parameter", "prior distribution", "posterior distribution"), 
    name = ""
  ) + # (凡例表示用)
  guides(
    linetype = guide_legend(override.aes = list(linewidth = 0.5), order = 1), 
    color    = guide_colorbar(order = 2), 
    fill     = guide_legend(order = 3)
  ) + 
  labs(
    title = "Gaussian-Gamma distribution", 
    subtitle = posterior_param_lbl, 
    fill  = "density", 
    color = "density", 
    x = expression(mu), 
    y = expression(lambda)
  )


### 予測分布(スチューデントのt分布)を計算 -----

#### パラメータの計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_s_hat     <- m_hat
lambda_s_hat <- beta_hat / (1 + beta_hat) * a_hat / b_hat
nu_s_hat     <- 2 * a_hat

mu_s_hat     <- (sum(x_n) + beta * m) / (N + beta)
term_1       <- (N + beta) / (N + 1 + beta)
term_2       <- (0.5 * N + a)
term_3       <- 0.5 * (sum(x_n^2) + beta * m^2 - 1/(N + beta) * (sum(x_n) + beta * m)^2) + b
lambda_s_hat <- term_1 * term_2 / term_3
nu_s_hat     <- N + 2 * a


#### 分布の計算 -----

# 予測分布の確率密度を計算
predict_df <- tibble::tibble(
  x        = x_vec, # 確率変数
  mu_s     = mu_s_hat,         # 位置パラメータ
  lambda_s = lambda_s_hat,     # 逆尺度パラメータ
  sigma_s  = 1/sqrt(lambda_s), # 尺度パラメータ
  nu_s     = nu_s_hat,         # 自由度パラメータ
  dens     = LaplacesDemon::dst(x = x, mu = mu_s, sigma = sigma_s, nu = nu_s) # 確率密度
)


#### 分布の作図 -----

# 予測分布のラベルを作成
predict_param_lbl <- paste0(
  "list(", 
  "N == ",              N, ", ", 
  "mu[truth] == ",      round(mu_truth,     digits = 2), ", ", 
  "lambda[truth] == ",  round(lambda_truth, digits = 5), ", ", 
  "hat(mu)[s] == ",     round(mu_s_hat,     digits = 2), ", ", 
  "hat(lambda)[s] == ", round(lambda_s_hat, digits = 5), ", ", 
  "hat(nu)[s] == ",     round(nu_s_hat,     digits = 1), 
  ")"
) |> 
  parse(text = _)

# 予測分布を作図
ggplot() + 
  geom_line(
    data    = model_df, 
    mapping = aes(x = x, y = dens, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_line(
    data    = predict_df, 
    mapping = aes(x = x, y = dens, color = "predict"), 
    linewidth = 1
  ) + # 予測分布
  scale_color_manual(
    breaks = c("model", "predict"), 
    values = c("red", "purple"), 
    labels = c("true model", "predict distribution"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5))
  ) + 
  labs(
    title = "Student's t distribution", 
    subtitle = predict_param_lbl, 
    x = expression(x), 
    y = "density"
  )


