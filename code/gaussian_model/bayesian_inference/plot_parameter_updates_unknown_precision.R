
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.2
# 精度が未知の場合
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)
library(LaplacesDemon)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

# 既知のパラメータを指定
mu <- 5

# 真のパラメータを指定
lambda_truth <- 0.25

# 標準偏差パラメータに変換
sigma_truth <- 1/sqrt(lambda_truth) # (処理の効率化用)
sigma_truth


### 観測データの設定 -----

# シードを設定:(ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# 観測データを生成
x_n <- rnorm(n = N, mean = mu, sd = 1/sqrt(lambda_truth))


### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_size <- sigma_truth |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  (\(.) {max(., abs(x_n-mu))})() |> # # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu - x_size
x_max <- mu + x_size
cat('x size:', x_min, x_max)

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, length.out = 1001)


# λ軸の範囲を設定
k <- 5
u <- 0.5
lambda_min <- 0
lambda_max <- lambda_truth |> # 真のパラメータ
  (\(.) {. * k})() |> # 定数倍
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
cat('λ size:', lambda_min, lambda_max)

# λ軸の値を作成
lambda_vec <- seq(from = lambda_min, to = lambda_max, length.out = 1001)


# σ軸の範囲を設定
sigma_min <- x_min - mu # (固定)
sigma_max <- x_max - mu # (固定)

# σ軸の値を作成
sigma_vec <- seq(from = sigma_min, to = sigma_max, length.out = 1001)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
a <- 1
b <- 1

# 予測分布のパラメータを計算:式(3.79)
mu_s     <- mu
lambda_s <- a / b
nu_s     <- 2 * a

# 受け皿を初期化
trace_a_i        <- rep(NA, times = N+1)
trace_b_i        <- rep(NA, times = N+1)
trace_mu_s_i     <- rep(NA, times = N+1)
trace_lambda_s_i <- rep(NA, times = N+1)
trace_nu_s_i     <- rep(NA, times = N+1)

# 初期値を記録
trace_a_i[1]        <- a
trace_b_i[1]        <- b
trace_mu_s_i[1]     <- mu_s
trace_lambda_s_i[1] <- lambda_s
trace_nu_s_i[1]     <- nu_s

# ベイズ推論による更新
for(n in 1:N){
  
  # 観測データを取得
  x <- x_n[n]
  
  # 事後分布のパラメータを更新:式(3.69)
  a <- a + 0.5
  b <- b + 0.5 * (x - mu)^2
  
  # 予測分布のパラメータを更新:式(3.79)
  mu_s     <- mu
  lambda_s <- a / b
  nu_s     <- 2 * a
  
  # 更新値を記録
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
a <- 1
b <- 1

# 事後分布のパラメータを計算:式(3.69)
trace_a_i <- 0.5 * 0:N + a
trace_b_i <- 0.5 * c(0, cumsum((x_n - mu)^2)) + b

# 予測分布のパラメータを計算:式(3.79)
trace_mu_s_i     <- rep(mu, times = N+1)
trace_lambda_s_i <- trace_a_i / trace_b_i
trace_nu_s_i     <- 0:N + 2 * trace_a_i


### 推移の作図 -----

#### 分布の計算 -----

# サンプルデータを格納
anim_sample_df <- tibble::tibble(
  n = 0:N,        # データ番号
  x = c(NA, x_n), # 観測値
  z = (sigma_truth / (x - mu))^2 # λ軸に対応する位置
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
    z = (sigma_truth / (x - mu))^2 # λ軸に対応する位置
  ) |> 
  tidyr::complete(
    i = 0:N, 
    fill = list(n = NA, x = NA, z = NA)
  ) |> # 初回フレーム用の値を追加
  dplyr::select(n = i, x, z)

# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu,             # 平均パラメータ
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
  lambda = lambda_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    a    = trace_a_i[n+1], # 形状パラメータ
    b    = trace_b_i[n+1], # 尺度パラメータ
    dens = dgamma(x = lambda, shape = a, rate = b) # 確率密度
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


#### 事後分布の作図 -----

# 事後分布のラベルを作成
anim_param_df <- tibble::tibble(
  n = 0:N, 
  a = trace_a_i, 
  b = trace_b_i, 
  posterior_param_lbl = sprintf(
    fmt = "list(N == '%s', lambda[truth] == %s, hat(a) == '%s', hat(b) == '%s')", 
    formatC(n,            digits = 0, format = "d"), 
    formatC(lambda_truth, digits = 5, format = "f"), 
    formatC(a,            digits = 1, format = "f"), 
    formatC(b,            digits = 1, format = "f")
  )
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_vline(
    mapping = aes(xintercept = lambda_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = anim_posterior_df, 
    mapping = aes(x = lambda, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = z, y = -Inf), 
    na.rm = TRUE, 
    color = "hotpink", alpha = 0.33, size = 2.5
  ) + # 観測データ
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = z, y = -Inf, color = "sample"), 
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
      breaks    = lambda_truth, 
      labels    = expression(lambda[truth])
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", "purple", "hotpink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name   = ""
  ) + # (凡例の表示用)
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
    xlim = c(lambda_min, lambda_max), # 描画範囲を固定
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "Gamma distribution", 
    subtitle = "", # (パラメータラベル用の空行)
    x = "λ", 
    y = "density"
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(
    file = "figure/gaussian_model/parameter_updates_precision/posterior.mp4"
  )
)


#### 予測分布の作図 -----

# 予測分布のラベルを作成
anim_param_df <- tibble::tibble(
  n        = 0:N, 
  mu_s     = trace_mu_s_i, 
  lambda_s = trace_lambda_s_i, 
  nu_s     = trace_nu_s_i, 
  predict_param_lbl = sprintf(
    fmt = "list(N == '%s', mu == %s, lambda[truth] == %s, mu[s] == '%s', hat(lambda)[s] == '%s', hat(nu)[s] == '%s')", 
    formatC(n,            digits = 0, format = "d"), 
    formatC(mu,           digits = 2, format = "f"), 
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
    file = "figure/gaussian_model/parameter_updates_precision/predict.mp4"
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


# p(λ)軸の範囲を設定
u <- 0.5
posterior_dens_max <- anim_posterior_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
posterior_dens_max

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
  dens_max = dnorm(x = mu, mean = mu, sd = sigma) # 最頻値における確率密度
)


# 第2軸の設定用のダミーを作成:(目盛の共通化用)
dummy_graph <- ggplot() + 
  geom_line(data = model_df, mapping = aes(x = x, y = dens)) + # 真の分布
  coord_cartesian(ylim = c(0, predict_dens_max)) # (目盛の共通化用)
dummy_built <- ggplot_build(dummy_graph) # 図情報を取得


# 試行ごとに作図
for(n in 0:N) {
  
  ##### パラメータの取得 -----
  
  # 値を取得
  i <- n + 1  # 試行インデックス
  x <- x_n[n] # 観測値
  a        <- trace_a_i[i]        # 形状パラメータ
  b        <- trace_b_i[i]        # 尺度パラメータ
  mu_s     <- trace_mu_s_i[i]     # 位置パラメータ
  lambda_s <- trace_lambda_s_i[i] # 逆尺度パラメータ
  sigma_s  <- 1/sqrt(lambda_s)    # 尺度パラメータ:(処理の効率化用)
  nu_s     <- trace_nu_s_i[i]     # 自由度パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x = x, # 観測値
    z = (sigma_truth / (x - mu))^2 # λ軸に対応する位置
  )
  data_df <- tibble::tibble(
    x = x_n[0:n], # 観測値
    z = (sigma_truth / (x - mu))^2 # λ軸に対応する位置
  )
  
  ##### 軸変換の作図：(σ to σ) -----
  
  # 恒等関数を作図
  identity_graph <- ggplot() + 
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
    ) + # (軸の対応用)
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
    ) + # (軸の対応用)
    labs(
      x = expression(lambda == frac(1, sigma^2)), 
      y = expression(sigma == frac(1, sqrt(lambda)))
    )
  
  ##### 観測データの作図 -----
  
  # 生成分布の期待値を計算
  E_x <- mu
  
  # 観測データの標本平均を計算
  bar_x = mean(x_n[0:n])
  
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
    fmt = "list(N == '%s', mu == %s, lambda[truth] == %s, paste(E(x) == mu, {} == '%s'), bar(x) == '%s')", 
    formatC(n,            digits = 0, format = "d"), 
    formatC(mu,           digits = 2, format = "f"), 
    formatC(lambda_truth, digits = 3, format = "f"), 
    formatC(E_x,          digits = 2, format = "f"), 
    formatC(bar_x,        digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 軸目盛を設定:(目盛の共通化用)
  dens_vals <- dummy_built$layout$panel_params[[1]]$y$breaks              # 確率密度軸目盛を取得
  freq_vals <- dens_vals * ifelse(test = n>0, yes = bin_size*n, no = 1) # 度数軸目盛に変換
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = c(mu+sigma_truth, mu)), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真・既知のパラメータ
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
        breaks    = c(mu+sigma_truth, E_x, bar_x), 
        labels    = c(expression(mu+sigma[truth]), expression(E(x)), expression(bar(x)))
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
  
  ##### 事後分布の作図 -----
  
  # 事後分布の期待値を計算
  E_lambda <- a / b
  
  # 事後分布の確率を計算
  posterior_df <- tibble::tibble(
    lambda = lambda_vec, # 確率変数
    dens   = dgamma(x = lambda, shape = a, rate = b) # 確率密度
  )
  
  # 事後分布のラベルを作成
  posterior_param_lbl <- sprintf(
    fmt = "list(hat(a) == '%s', hat(b) == '%s', paste(E(lambda) == frac(hat(a), hat(b)), {} == '%s'))", 
    formatC(a,        digits = 1, format = "f"), 
    formatC(b,        digits = 1, format = "f"), 
    formatC(E_lambda, digits = 3, format = "f")
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = lambda_truth, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_lambda), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_line(
      data    = posterior_df, 
      mapping = aes(x = lambda, y = dens, color = "posterior"), 
      linewidth = 1
    ) + # 事後分布
    geom_point(
      data    = data_df, 
      mapping = aes(x = z, y = 0), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = z, y = 0), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(lambda_truth, E_lambda), 
        labels    = c(expression(lambda[truth]), expression(E(lambda)))
      ) # パラメータラベル
    ) + 
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
      color = guide_legend(override.aes = list(linewidth = 0.5))
    ) + 
    coord_cartesian(
      xlim = c(lambda_min, lambda_max),         # (目盛の共通化用)
      ylim = c(0, posterior_dens_max) # 表示範囲を固定
    ) + 
    labs(
      title = "Gamma distribution", 
      subtitle = posterior_param_lbl, 
      x = expression(lambda), 
      y = "density"
    )
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x <- mu_s
  
  # 最頻値における確率密度を計算
  N_dens_max   <- dnorm(x = mu, mean = mu, sd = sigma_truth)
  t_dens_max   <- LaplacesDemon::dst(x = mu_s, mu = mu_s, sigma = sigma_s, nu = nu_s)
  
  # 予測分布の確率密度を計算
  predict_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    dens = LaplacesDemon::dst(x = x, mu = mu_s, sigma = sigma_s, nu = nu_s) # 確率密度
  )
  
  # 予測分布のラベルを作成
  predict_param_lbl <- sprintf(
    fmt = "list(mu[s] == '%s', hat(lambda)[s] == '%s', hat(nu)[s] == '%s', paste(E(x) == mu[s], {} == '%s'))", 
    formatC(mu_s,     digits = 2, format = "f"), 
    formatC(lambda_s, digits = 3, format = "f"), 
    formatC(nu_s,     digits = 1, format = "f"), 
    formatC(E_x,      digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 予測分布を作図
  predict_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 既知のパラメータ
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
        breaks    = c(mu, E_x), 
        labels    = c(expression(mu), expression(E(x)))
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
  N_dens_max   <- dnorm(x = mu, mean = mu, sd = sigma_truth)
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
        expression(St(x == mu[s] ~'|'~ mu[s], lambda, hat(nu)[s]))
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
    ) + # (軸の対応用)
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
    identity_graph, adapt_s2l_graph, 
    model_graph,    posterior_graph, 
    predict_graph,  adapt_l2p_graph, 
    nrow = 3, ncol = 2, 
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
    width = 12, height = 15, units = "in", dpi = 100
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
    path = "figure/gaussian_model/parameter_updates_precision/observation.mp4", 
    framerate = 10 
  ) -> tmp_path # mp4ファイルを書出


