
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(LaplacesDemon)
library(gganimate)

# パッケージの省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布の設定 -----

# 真のパラメータを指定
mu_truth     <- 5
lambda_truth <- 0.25

# 標準偏差パラメータに変換
sigma_truth <- 1/sqrt(lambda_truth) # (処理の効率化用)
sigma_truth


### 観測データの生成 -----

# シードを設定:(ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# 観測データを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sigma_truth)


### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_size <- sigma_truth |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  (\(.) {max(., abs(x_n-mu_truth))})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu_truth - x_size
x_max <- mu_truth + x_size
cat("x size:", x_min, x_max)

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, length.out = 101)


# μ軸の範囲を設定
mu_min <- x_min # (固定)
mu_max <- x_max # (固定)
cat("μ size:", mu_min, mu_max)

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 101)


# λ軸の範囲を設定
k <- 5
u <- 0.5
lambda_min <- 0
lambda_max <- lambda_truth |> # 真のパラメータ
  (\(.) {. * k})() |> # 定数倍
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
cat("λ size:", lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 101)


# σ軸の範囲を設定
sigma_min <- x_min - mu_truth # (固定)
sigma_max <- x_max - mu_truth # (固定)
cat("σ size:", sigma_min, sigma_max)

# σ軸の値を作成
sigma_vec <- seq(from = sigma_min, to = sigma_max, length.out = 1001)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを指定
m    <- 0
beta <- 1
a    <- 1
b    <- 1

# 予測分布のパラメータを計算:式(3.95)
mu_s     <- m
lambda_s <- beta / (1 + beta) * a / b
nu_s     <- 2 * a

# 受け皿を初期化
trace_m_i        <- rep(NA, times = N+1)
trace_beta_i     <- rep(NA, times = N+1)
trace_a_i        <- rep(NA, times = N+1)
trace_b_i        <- rep(NA, times = N+1)
trace_mu_s_i     <- rep(NA, times = N+1)
trace_lambda_s_i <- rep(NA, times = N+1)
trace_nu_s_i     <- rep(NA, times = N+1)

# 初期値を記録
trace_m_i[1]        <- m
trace_beta_i[1]     <- beta
trace_a_i[1]        <- a
trace_b_i[1]        <- b
trace_mu_s_i[1]     <- mu_s
trace_lambda_s_i[1] <- lambda_s
trace_nu_s_i[1]     <- nu_s

# ベイズ推論による更新
for(n in 1:N){
  
  # 観測データを取得
  x <- x_n[n]
  
  # μの事後分布のパラメータを更新:式(3.83)
  beta_old <- beta
  m_old    <- m
  beta     <- 1 + beta
  m        <- (x_n[n] + beta_old * m) / beta
  
  # λの事後分布のパラメータを更新:式(3.88)
  a <- 0.5 + a
  b <- 0.5 * (x_n[n]^2 + beta_old * m_old^2 - beta * m^2) + b
  
  # 予測分布のパラメータを更新:式(3.95)
  mu_s     <- m
  lambda_s <- beta / (1 + beta) * a / b
  nu_s     <- 2 * a
  
  # 更新値を記録
  trace_m_i[n+1]        <- m
  trace_beta_i[n+1]     <- beta
  trace_a_i[n+1]        <- a
  trace_b_i[n+1]        <- b
  trace_mu_s_i[n+1]     <- mu_s
  trace_lambda_s_i[n+1] <- lambda_s
  trace_nu_s_i[n+1]     <- nu_s
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m    <- 0
beta <- 1
a    <- 1
b    <- 1




# 試行ごとに事後分布と予測分布のパラメータを計算
anime_param_df <- tibble::tibble(
  # 計算用の値
  n = 0:N, 
  sum_x = cumsum(c(0, x_n)), 
  sum_x2 = cumsum(c(0, x_n)^2), 
  # μの事後分布のパラメータ:式(3.83)
  beta_hat = n + beta, 
  m_hat = (sum_x + beta*m) / (n + beta), 
  # λの事後分布のパラメータ:式(3.88)
  a_hat = 0.5*n + a, 
  b_hat = 0.5 * (sum_x2 + beta*m^2 - beta_hat*m_hat^2) + b, 
  # 予測分布のパラメータ:式(3.95)
  mu_st_hat = m_hat, 
  lambda_st_hat = beta_hat * a_hat / (1 + beta_hat) / b_hat, 
  nu_st_hat = 2 * a_hat
) |> # パラメータを計算
  dplyr::select(!c(sum_x, sum_x2)) # 不要な列を削除


### 分布の計算 -----

# サンプルデータを格納
anim_sample_df <- tibble::tibble(
  n = 0:N,        # データ番号
  x = c(NA, x_n), # 観測値
  z = (sigma_truth / (x - mu_truth))^2 # λ軸に対応する座標
) |> 
  dplyr::mutate(
    z = dplyr::if_else(
      condition = z <= lambda_max, 
      true      = z, 
      false     = NA
    ) # 描画範囲外を非表示化
  )

# 観測データを格納
anim_data_df <- tibble::tibble(
  i = 1:N # フレーム番号
) |> 
  dplyr::reframe(
    n = 1:i, # データ番号
    .by = i
  ) |> 
  dplyr::mutate(
    x = x_n[n], # 観測値
    z = (sigma_truth / (x - mu_truth))^2 # λ軸に対応する座標
  ) |> 
  dplyr::mutate(
    z = dplyr::if_else(
      condition = z <= lambda_max, 
      true      = z, 
      false     = NA
    ) # 描画範囲外を非表示化
  ) |> 
  tidyr::complete(
    i = 0:N, 
    fill = list(n = NA, x = NA, z = NA)
  ) |> # 初期値用のデータを補完
  dplyr::select(n = i, x, z)


# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu_truth,       # 平均パラメータ
  lambda = lambda_truth,   # 精度パラメータ
  sigma  = 1/sqrt(lambda), # 標準偏差パラメータ
  dens   = dnorm(x = x, mean = mu, sd = sigma) # 確率密度
)

# 生成分布を複製
anim_model_df <- tidyr::expand_grid(
  n = 0:N, # データ番号
  model_df
) # 試行ごとに分布を複製


# 事後分布の確率密度を計算
anim_posterior_df <- tidyr::expand_grid(
  n      = 0:N,       # 試行回数
  mu     = mu_vec,    # μの確率変数
  lambda = lambda_vec # λの確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    m         = trace_m_i[n+1],    # 平均パラメータ
    beta      = trace_beta_i[n+1], # 係数パラメータ
    lambda_mu = beta * lambda,     # 精度パラメータ
    sigma_mu  = 1/sqrt(lambda_mu), # 標準偏差パラメータ
    a         = trace_a_i[n+1],    # 形状パラメータ
    b         = trace_b_i[n+1],    # 尺度パラメータ
    N_dens    = dnorm(x = mu, mean = m, sd = sigma_mu),  # μの確率密度
    Gam_dens  = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
    dens      = N_dens * Gam_dens # μ, λの確率密度
  )

# 予測分布の確率密度を計算
anim_predict_df <- tidyr::expand_grid(
  n = 0:N,  # 試行回数
  x = x_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    mu_s     = trace_mu_s_i[n+1],     # 位置パラメータ
    lambda_s = trace_lambda_s_i[n+1], # 逆尺度パラメータ
    sigma_s  = 1/sqrt(lambda_s),      # 尺度パラメータ
    nu_s     = trace_nu_s_i[n+1],     # 自由度パラメータ
    dens     = LaplacesDemon::dst(x = x, mu = mu_s, sigma = sigma_s, nu = nu_s) # 確率密度
  )


### 推移の作図 -----

#### 事後分布の作図 -----

# 事後分布のラベルを作成
anim_param_df <- tibble::tibble(
  n    = 0:N, 
  m    = trace_m_i, 
  beta = trace_beta_i, 
  a    = trace_a_i, 
  b    = trace_b_i
) |> 
  dplyr::mutate(
    posterior_param_lbl = sprintf(
      fmt = paste0(
        "list(", 
        "N == '%s', mu[truth] == %s, lambda[truth] == %s, ", 
        "hat(m) == '%s', hat(beta) == '%s', hat(a) == '%s', hat(b) == '%s'", 
        ")"
      ), 
      formatC(n,            digits = 0, format = "d"), 
      formatC(mu_truth,     digits = 2, format = "f"), 
      formatC(lambda_truth, digits = 5, format = "f"), 
      formatC(m,            digits = 2, format = "f"), 
      formatC(beta,         digits = 1, format = "f"), 
      formatC(a,            digits = 1, format = "f"), 
      formatC(b,            digits = 1, format = "f")
    )
  )

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_contour_filled(
    data    = anim_posterior_df, 
    mapping = aes(x = mu, y = lambda, z = dens, fill = after_stat(level), color = "posterior"), 
    alpha = 0.5, linetype = "blank"
  ) + # 事後分布
  geom_vline(
    mapping = aes(xintercept = mu_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_hline(
    mapping = aes(yintercept = lambda_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x, y = z), 
    na.rm = TRUE, 
    color = "hotpink", alpha = 0.33, size = 2.5
  ) + # 観測データ
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x, y = z, color = "sample"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = posterior_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -1
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
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
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", "black", "hotpink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name   = ""
  ) + # (凡例の表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5), order = 1), 
    fill  = guide_legend(order = 2)
  ) + 
  theme(
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    xlim = c(mu_min, mu_max),         # 描画範囲を固定
    ylim = c(lambda_min, lambda_max), # 描画範囲を固定
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Gaussian-Gamma distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    fill  = "density", 
    x = "μ", 
    y = "λ"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(
    file = "figure/gaussian_model/parameter_updates_mean_precision/posterior.mp4"
  )
)


#### 予測分布の作図 -----

# 予測分布のラベルを作成
anim_param_df <- tibble::tibble(
  n        = 0:N, 
  mu_s     = trace_mu_s_i, 
  lambda_s = trace_lambda_s_i, 
  nu_s     = trace_nu_s_i
) |> 
  dplyr::mutate(
    predict_param_lbl = sprintf(
      fmt = paste0(
        "list(", 
        "N == '%s', mu[truth] == %s, lambda[truth] == %s, ", 
        "hat(mu)[s] == '%s', hat(lambda)[s] == '%s', hat(nu)[s] == '%s'", 
        ")"
      ), 
      formatC(n,            digits = 0, format = "d"), 
      formatC(mu_truth,     digits = 2, format = "f"), 
      formatC(lambda_truth, digits = 5, format = "f"), 
      formatC(mu_s,         digits = 2, format = "f"), 
      formatC(lambda_s,     digits = 5, format = "f"), 
      formatC(nu_s,         digits = 1, format = "f")
    )
  )

# 予測分布を作図
posterior_graph <- ggplot() + 
  geom_line(
    data    = anim_model_df, 
    mapping = aes(x = x, y = dens, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_line(
    data    = anim_predict_df, 
    mapping = aes(x = x, y = dens, color = "predict"), 
    linewidth = 1
  ) + # 予測分布
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x, y = -Inf), 
    na.rm = TRUE, 
    color = "hotpink", alpha = 0.33, size = 2.5
  ) + # 観測データ
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x, y = -Inf, color = "sample"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = predict_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -0.5
  ) + # パラメータラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_color_manual(
    breaks = c("model", "predict", "sample"), 
    values = c("red", "purple", "hotpink"), 
    labels = c("true model", "predict distribution", "observation data"), 
    name   = ""
  ) + # (凡例表示用)
  guides(
    color = guide_legend(
      override.aes = list(
        linewidth = c(0.5, 0.5, NA), 
        linetype  = c("dashed", "solid", NA), 
        size      = c(NA, NA, 5), 
        shape     = c(NA, NA, "circle"))
    )
  ) + 
  theme(
    legend.title           = element_text(size = 0), 
    legend.position        = "inside", 
    legend.position.inside = c(1, 1), 
    legend.justification   = c(1, 1), 
    legend.background      = element_rect(fill = alpha("white", alpha = 0.8)), 
    plot.subtitle = element_text(size = 50) # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Student's t Distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "μ", 
    y = "density"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(
    file = "figure/gaussian_model/parameter_updates_mean_precision/predict.mp4"
  )
)


#### 観測データと分布の関係 -----

# 一時書き出し先を指定
dir_path <- "figure/tmp_folder"


# 階級数を指定
bin_num <- 40

# 階級値を作成
bin_size   <- (x_max - x_min) / bin_num # 階級幅
bin_min    <- x_min - 0.5*bin_size # 境界値の最小値
bin_max    <- x_max + 0.5*bin_size # 境界値の最大値
center_vec <- seq(from = x_min, to = x_max, by = bin_size) # 階級値
center_vec


# p(μ, λ)軸の範囲を設定
u <- 1
posterior_dens_max <- anim_posterior_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
posterior_dens_max

# 等高線を設定
level_num <- 23
posterior_dens_vals <- seq(from = 0, to = posterior_dens_max, length.out = level_num)
posterior_dens_vals

# p(x)軸の範囲を設定
u <- 0.05
predict_dens_max <- anim_predict_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
predict_dens_max


# 恒等関数の座標を計算
identity_coord_df <- tibble::tibble(
  sigma = sigma_vec # 座標
)

# 変換曲線の座標を計算
adapt_coord_df <- tibble::tibble(
  lambda   = lambda_vec,     # 生成分布の確率変数
  sigma    = 1/sqrt(lambda), # 標準偏差
  dens_max = dnorm(x = mu_truth, mean = mu_truth, sd = sigma) # 最頻値における確率密度
)


# 第2軸の設定用のダミーを作成:(目盛の共通化用)
dummy_graph <- ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens)) + # 真の分布
  coord_cartesian(ylim = c(0, predict_dens_max)) # (目盛の共通化用)
dummy_built <- ggplot_build(dummy_graph) # 図情報を取得

n <- 0
# 試行ごとに作図
for(n in 0:N) {
  
  ##### パラメータの取得 -----
  
  # 値を取得
  i <- n + 1  # 試行インデックス
  x <- x_n[n] # 観測値
  m        <- trace_m_i[i]        # 平均パラメータ
  beta     <- trace_beta_i[i]     # 係数パラメータ
  a        <- trace_a_i[i]        # 形状パラメータ
  b        <- trace_b_i[i]        # 尺度パラメータ
  mu_s     <- trace_mu_s_i[i]     # 位置パラメータ
  lambda_s <- trace_lambda_s_i[i] # 逆尺度パラメータ
  sigma_s  <- 1/sqrt(lambda_s)    # 尺度パラメータ:(処理の効率化用)
  nu_s     <- trace_nu_s_i[i]     # 自由度パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x = x, # 観測値
    z = (sigma_truth / (x - mu_truth))^2 # λ軸に対応する座標
  )
  data_df <- tibble::tibble(
    x = x_n[0:n], # 観測値
    z = (sigma_truth / (x - mu_truth))^2 # λ軸に対応する座標
  )
  
  ##### 軸変換の作図：(σ to σ) -----
  
  # 恒等関数を作図
  identity_s2s_graph <- ggplot() + 
    geom_line(
      data    = identity_coord_df, 
      mapping = aes(x = sigma, y = sigma), 
      linewidth = 1
    ) + # 恒等関数
    geom_segment(
      mapping = aes(
        x    = sigma_truth, 
        y    = c(-Inf, sigma_truth), 
        xend = c(sigma_truth, Inf), 
        yend = sigma_truth
      ), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    scale_y_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = sigma_truth, 
        labels    = expression(sigma[truth])
      ) # パラメータラベル
    ) + 
    coord_cartesian(
      xlim = c(sigma_min, sigma_max), # (目盛の共通化用)
      ylim = c(sigma_min, sigma_max)  # (目盛の共通化用)
    ) + 
    labs(
      x = expression(sigma), 
      y = expression(sigma)
    )
  
  ##### 軸変換の作図：(σ to λ) -----
  
  # 変換曲線を作図
  adapt_s2l_graph <- ggplot() + 
    geom_line(
      data    = adapt_coord_df, 
      mapping = aes(x = lambda, y = sigma), 
      linewidth = 1
    ) + # 変換曲線
    geom_segment(
      mapping = aes(
        x    = c(-Inf, lambda_truth), 
        y    = sigma_truth, 
        xend = lambda_truth, 
        yend = c(sigma_truth, -Inf)
      ), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    coord_cartesian(
      xlim = c(lambda_min, lambda_max), # (目盛の共通化用)
      ylim = c(sigma_min, sigma_max)    # (目盛の共通化用)
    ) + 
    labs(
      x = expression(lambda == frac(1, sigma^2)), 
      y = expression(sigma == frac(1, sqrt(lambda)))
    )
  
  ##### 観測データの作図 -----
  
  # 生成分布の期待値を計算
  E_x <- mu_truth
  
  # 観測データの標本平均を計算
  bar_x <- mean(x_n[0:n])
  
  # 観測データを集計
  obs_df <- data_df |> 
    dplyr::mutate(
      bin_i  = (x - bin_min) %/% bin_size,        # 階級番号
      center = bin_min + (bin_i + 0.5) * bin_size # 階級値
    ) |> 
    dplyr::count(
      center, name = "freq" # 度数
    ) |> 
    dplyr::mutate(
      dens = freq / (bin_size * n) # 密度
    ) |> 
    tidyr::complete(
      center = center_vec, 
      fill = list(freq = 0, dens = 0)
    ) # 未観測値を補完
  
  # 生成分布のラベルを作成
  model_param_lbl <- sprintf(
    fmt = paste0(
      "list(", 
      "N == '%s', mu[truth] == %s, lambda[truth] == %s, ", 
      "paste(E(x) == mu[truth], {} == '%s'), bar(x) == '%s'", 
      ")"
    ), 
    formatC(n,            digits = 0, format = "d"), 
    formatC(mu_truth,     digits = 2, format = "f"), 
    formatC(lambda_truth, digits = 3, format = "f"), 
    formatC(E_x,          digits = 2, format = "f"), 
    formatC(bar_x,        digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 軸目盛を設定:(目盛の共通化用)
  dens_vals <- dummy_built$layout$panel_params[[1]]$y$breaks            # 確率密度軸目盛を取得
  freq_vals <- dens_vals * ifelse(test = n>0, yes = bin_size*n, no = 1) # 度数軸目盛に変換
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = c(mu_truth+sigma_truth, mu_truth)), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = bar_x), 
      color = "hotpink", linewidth = 1, linetype = "dashed"
    ) + # 標本平均
    geom_line(
      data    = model_df, 
      mapping = aes(x = x, y = dens, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真の分布
    geom_bar(
      data    = obs_df, 
      mapping = aes(x = center, y = dens, color = "sample"), 
      stat = "identity", position = "identity", width = bin_size, 
      fill = "hotpink", alpha = 0.5, linetype = "blank"
    ) + # 観測データ
    geom_point(
      data    = data_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth+sigma_truth, E_x, bar_x), 
        labels    = c(expression(mu[truth]+sigma[truth]), expression(E(x)), expression(bar(x)))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      breaks = dens_vals, # (目盛の共通化用)
      sec.axis = sec_axis(
        transform = ~ . * ifelse(test = n>0, yes = bin_size*n, no = 1), 
        breaks    = freq_vals, # (目盛の共通化用)
        labels    = scales::label_number(accuracy = 0.01), # (描画領域のズレの対策用)
        name      = "frequency"
      ) # 度数軸目盛
    ) + 
    scale_color_manual(
      breaks = c("model", "sample"), 
      values = c("red", NA), 
      labels = c("true model", "observation data"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(x_min, x_max),       # (目盛の共通化用)
      ylim = c(0, predict_dens_max) # (目盛の共通化用)
    ) + 
    labs(
      title = "Gaussian distribution", 
      subtitle = model_param_lbl, 
      x = expression(x), 
      y = "density"
    )
  
  ##### λ軸の作図 -----
  
  # λ軸を作図
  l_axis_graph <- ggplot() + 
    geom_vline(
      xintercept = lambda_truth, 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    coord_cartesian(
      xlim = c(lambda_min, lambda_max), # (目盛の共通化用)
    ) + 
    labs(
      x = expression(lambda)
    )
  
  ##### 事後分布の作図 -----
  
  # 事後分布の期待値を計算
  E_mu     <- m
  E_lambda <- a / b
  
  # 事後分布の確率を計算
  posterior_df <- tidyr::expand_grid(
    mu     = mu_vec,    # μの確率変数
    lambda = lambda_vec # λの確率変数
  ) |> # 格子点を作成
    dplyr::mutate(
      N_dens   = dnorm(x = mu, mean = m, sd = 1/sqrt(beta*lambda)),  # μの確率密度
      Gam_dens = dgamma(x = lambda, shape = a, rate = b), # λの確率密度
      dens     = N_dens * Gam_dens # μ, λの確率密度
    )
  
  # 事後分布のラベルを作成
  posterior_param_lbl <- sprintf(
    fmt = paste0(
      "list(", 
      "hat(m) == '%s', hat(beta) == '%s', hat(a) == '%s', hat(b) == '%s', ", 
      "paste(E(mu) == hat(m), {} == '%s'), paste(E(lambda) == frac(hat(a), hat(b)), {} == '%s')", 
      ")"
    ), 
    formatC(m,        digits = 2, format = "f"), 
    formatC(beta,     digits = 1, format = "f"), 
    formatC(a,        digits = 1, format = "f"), 
    formatC(b,        digits = 1, format = "f"), 
    formatC(E_mu,     digits = 2, format = "f"), 
    formatC(E_lambda, digits = 3, format = "f")
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_contour_filled(
      data    = posterior_df, 
      mapping = aes(x = mu, y = lambda, z = dens, fill = after_stat(level), color = "posterior"), 
      breaks = posterior_dens_vals, # (目盛の共通化用)
      alpha = 0.5, linetype = "blank"
    ) + # 事後分布
    geom_vline(
      mapping = aes(xintercept = mu_truth, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_hline(
      mapping = aes(yintercept = lambda_truth, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      xintercept = E_mu, 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_hline(
      yintercept = E_lambda, 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_point(
      data    = data_df, 
      mapping = aes(x = x, y = z), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = z), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth, E_mu), 
        labels    = c(expression(mu[truth]), expression(E(mu)))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(lambda_truth, E_lambda), 
        labels    = c(expression(lambda[truth]), expression(E(lambda)))
      ) # パラメータラベル
    ) + 
    scale_fill_viridis_d(drop = FALSE) + # (目盛の固定用)
    scale_color_manual(
      breaks = c("model", "posterior"), 
      values = c("red", "purple"), 
      labels = c("true parameter", "posterior distribution"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(override.aes = list(linewidth = 0.5)), 
      fill  = "none"
    ) + 
    coord_cartesian(
      xlim = c(mu_min, mu_max),        # (目盛の共通化用)
      ylim = c(lambda_min, lambda_max) # (目盛の共通化用)
    ) + 
    labs(
      title = "Gamma distribution", 
      subtitle = posterior_param_lbl, 
      fill = "density", 
      x = expression(lambda), 
      y = "density"
    )
  
  ##### 軸変換の作図：(λ to λ) -----
  
  # 恒等関数を作図
  identity_l2l_graph <- ggplot() + 
    geom_line(
      data    = adapt_coord_df, 
      mapping = aes(x = lambda, y = lambda), 
      linewidth = 1
    ) + # 恒等関数
    geom_vline(
      xintercept = lambda_truth, 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_segment(
      mapping = aes(
        x    = c(-Inf, lambda_truth), 
        y    = lambda_truth, 
        xend = lambda_truth, 
        yend = c(lambda_truth, Inf)
      ), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_segment(
      mapping = aes(
        x    = c(-Inf, E_lambda), 
        y    = E_lambda, 
        xend = E_lambda, 
        yend = c(E_lambda, -Inf)
      ), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    coord_cartesian(
      xlim = c(lambda_min, lambda_max), # (目盛の共通化用)
      ylim = c(lambda_min, lambda_max)  # (目盛の共通化用)
    ) + 
    labs(
      x = expression(lambda), 
      y = expression(lambda)
    )
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x <- mu_s
  
  # 最頻値における確率密度を計算
  N_dens_max   <- dnorm(x = mu_truth, mean = mu_truth, sd = sigma_truth)
  t_dens_max   <- LaplacesDemon::dst(x = mu_s, mu = mu_s, sigma = sigma_s, nu = nu_s)
  
  # 予測分布の確率密度を計算
  predict_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    dens = LaplacesDemon::dst(x = x, mu = mu_s, sigma = sigma_s, nu = nu_s) # 確率密度
  )
  
  # 予測分布のラベルを作成
  predict_param_lbl <- sprintf(
    fmt = "list(hat(mu)[s] == '%s', hat(lambda)[s] == '%s', hat(nu)[s] == '%s', paste(E(x) == hat(mu)[s], {} == '%s'))", 
    formatC(mu_s,     digits = 2, format = "f"), 
    formatC(lambda_s, digits = 3, format = "f"), 
    formatC(nu_s,     digits = 1, format = "f"), 
    formatC(E_x,      digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 予測分布を作図
  predict_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_hline(
      mapping = aes(yintercept = N_dens_max), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_x), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_hline(
      mapping = aes(yintercept = t_dens_max), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
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
    geom_point(
      data    = data_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth, E_x), 
        labels    = c(expression(mu[truth]), expression(E(x)))
      ) # パラメータラベル
    ) + 
    scale_color_manual(
      breaks = c("model", "predict"), 
      values = c("red", "purple"), 
      labels = c("true model", "predict distribution"), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      color = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(x_min, x_max),       # (目盛の共通化用)
      ylim = c(0, predict_dens_max) # 表示範囲を固定
    ) + 
    labs(
      title = "Student's t distribution", 
      subtitle = predict_param_lbl, 
      x = expression(x), 
      y = "density"
    )
  
  ##### 軸変換の作図：(λ to p(x)) -----
  
  # 最頻値における確率密度を計算
  N_dens_max   <- dnorm(x = mu_truth, mean = mu_truth, sd = sigma_truth)
  t_dens_max   <- LaplacesDemon::dst(x = mu_s, mu = mu_s, sigma = sigma_s, nu = nu_s)
  
  # 変換曲線の座標を計算
  tmp_coord_df <- tibble::tibble(
    lambda   = lambda_vec,     # 事後分布の確率変数
    sigma    = 1/sqrt(lambda), # 尺度パラメータ
    dens_max = LaplacesDemon::dst(x = mu_s, mu = mu_s, sigma = sigma, nu = nu_s) # 最頻値における確率密度
  )
  
  # 変換曲線を作図
  adapt_l2p_graph <- ggplot() + 
    geom_line(
      data    = adapt_coord_df, 
      mapping = aes(x = lambda, y = dens_max, linetype = "model"), 
      linewidth = 1
    ) + # 変換曲線:ガウス分布
    geom_line(
      data    = tmp_coord_df, 
      mapping = aes(x = lambda, y = dens_max, linetype = "predict"), 
      color = "purple", linewidth = 1
    ) + # 変換曲線:t分布
    geom_segment(
      mapping = aes(
        x    = lambda_truth, 
        y    = c(Inf, N_dens_max), 
        xend = c(lambda_truth, -Inf), 
        yend = N_dens_max
      ), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_segment(
      mapping = aes(
        x    = lambda_s, 
        y    = c(Inf, t_dens_max), 
        xend = c(lambda_s, -Inf), 
        yend = t_dens_max
      ), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(lambda_truth, lambda_s), 
        labels    = c(expression(lambda[truth]), expression(lambda[s]))
      ) # パラメータラベル
    ) + 
    scale_linetype_manual(
      breaks = c("model", "predict"), 
      values = c("solid", "dotdash"), 
      labels = c(
        expression(N(x == mu ~'|'~ mu, lambda^{-1})), 
        expression(St(x == mu[s] ~'|'~ hat(mu)[s], lambda, hat(nu)[s]))
      ), 
      name   = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 0), 
      legend.justification   = c(1, 0), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      linetype = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(lambda_min, lambda_max), # (目盛の共通化用)
      ylim = c(0, predict_dens_max)     # (目盛の共通化用)
    ) + 
    labs(
      x = expression(lambda), 
      y = "density"
    )
  
  ##### 対応関係の作図 -----
  
  # 図タイトルを指定
  title_graph <- cowplot::ggdraw() + 
    cowplot::draw_label(
      label = "Bayesian inference", 
      size = 20
    ) + 
    theme(
      plot.background = element_rect(fill = "white", color = NA) # (透過背景の対策用)
    )
  
  # グラフを並べて描画
  comb_graph <- cowplot::plot_grid(
    identity_s2s_graph, adapt_s2l_graph, 
    model_graph,        l_axis_graph, 
    posterior_graph,    identity_l2l_graph, 
    predict_graph,      adapt_l2p_graph, 
    nrow = 4, ncol = 2, 
    align = "hv" # (目盛の共通化用)
  )
  final_graph <- cowplot::plot_grid(
    title_graph, comb_graph, 
    nrow = 2, ncol = 1, 
    rel_heights = c(0.05, 1)
  )
  
  ##### グラフの出力 -----
  
  # 画像ファイルを書出
  file_path <- paste0(dir_path, "/", stringr::str_pad(n, width = nchar(N), pad = "0"), ".png")
  ggplot2::ggsave(
    filename = file_path, plot = final_graph, 
    width = 12, height = 19, units = "in", dpi = 100
  )
  
  # 途中経過を表示
  message("\r", n, " / ", N, appendLF = FALSE)
}


##### アニメーションの変換 -----

# 動画を作成
paste0(dir_path, "/", stringr::str_pad(0:N, width = nchar(N), pad = "0"), ".png") |> # ファイルパスを作成
  magick::image_read() |> # pngファイルを読込
  magick::image_animate(fps = 1, dispose = "previous") |> # gifファイルを作成
  magick::image_write_video(
    path = "figure/gaussian_model/parameter_updates_mean_precision/observation.mp4", 
    framerate = 10 
  ) -> tmp_path # mp4ファイルを書出


