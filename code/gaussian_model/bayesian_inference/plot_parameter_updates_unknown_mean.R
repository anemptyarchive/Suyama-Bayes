
# 1次元ガウスモデル ------------------------------------------------------------

# chapter 3.3.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布の設定 -----

# 真のパラメータを指定
mu_truth <- 25

# 既知のパラメータを指定
lambda <- 0.01
1/sqrt(lambda) # 標準偏差


### 観測データの生成 -----

# シードを設定:(ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# 観測データを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = 1/sqrt(lambda))


### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_size <- (1/sqrt(lambda)) |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  (\(.) {max(., abs(x_n-mu_truth))})() |> # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_min <- mu_truth - x_size
x_max <- mu_truth + x_size
cat("x size:", x_min, x_max)

# x軸の値を作成
x_vec <- seq(from = x_min, to = x_max, length.out = 1001)


# μ軸の範囲を設定
mu_min <- x_min # (固定)
mu_max <- x_max # (固定)
cat("μ size:", mu_min, mu_max)

# μ軸の値を作成
mu_vec <- seq(from = mu_min, to = mu_max, length.out = 1001)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
m         <- 0
lambda_mu <- 0.001

# 予測分布のパラメータを計算:式(3.62)
mu_star     <- m
lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)

# 受け皿を初期化
trace_m_i           <- rep(NA, times = N+1)
trace_lambda_mu_i   <- rep(NA, times = N+1)
trace_mu_star_i     <- rep(NA, times = N+1)
trace_lambda_star_i <- rep(NA, times = N+1)

# 初期値を記録
trace_m_i[1]           <- m
trace_lambda_mu_i[1]   <- lambda_mu
trace_mu_star_i[1]     <- mu_star
trace_lambda_star_i[1] <- lambda_star

# ベイズ推論による更新
for(n in 1:N){
  
  # 観測データを取得
  x <- x_n[n]
  
  # 事後分布のパラメータを更新:式(3.53, 3.54)
  lambda_mu_old <- lambda_mu
  lambda_mu     <- lambda_mu + lambda
  m             <- (x_n[n] * lambda + m * lambda_mu_old) / lambda_mu
  
  # 予測分布のパラメータを更新:式(3.62)
  mu_star     <- m
  lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
  
  # 更新値を記録
  trace_m_i[n+1]           <- m
  trace_lambda_mu_i[n+1]   <- lambda_mu
  trace_mu_star_i[n+1]     <- mu_star
  trace_lambda_star_i[n+1] <- lambda_star
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m         <- 0
lambda_mu <- 0.001

# 事後分布のパラメータを計算:式(3.53, 3.54)
trace_lambda_mu_i <- 0:N * lambda + lambda_mu
trace_m_i         <- (c(0, cumsum(x_n)) * lambda + m * lambda_mu) / trace_lambda_mu_i

# 予測分布のパラメータを計算:式(3.62)
trace_mu_star_i     <- trace_m_i
trace_lambda_star_i <- lambda * trace_lambda_mu_i / (lambda + trace_lambda_mu_i)


### 分布の計算 -----

# サンプルデータを格納
anim_sample_df <- tibble::tibble(
  n = 0:N,       # データ番号
  x = c(NA, x_n) # 観測値
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
    x = x_n[n] # 観測値
  ) |> 
  tidyr::complete(
    i = 0:N, 
    fill = list(n = NA, x = NA)
  ) |> # 初期値用のデータを補完
  dplyr::select(n = i, x)


# 生成分布の確率密度を計算
model_df <- tibble::tibble(
  x      = x_vec, # 確率変数
  mu     = mu_truth,       # 平均パラメータ
  lambda = lambda,         # 精度パラメータ
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
  n  = 0:N,   # 試行回数
  mu = mu_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    m         = trace_m_i[n+1],         # 平均パラメータ
    lambda_mu = trace_lambda_mu_i[n+1], # 精度パラメータ
    sigma_mu  = 1/sqrt(lambda_mu),      # 標準偏差パラメータ
    dens      = dnorm(x = mu, mean = m, sd = sigma_mu) # 確率密度
  )

# 予測分布の確率密度を計算
anim_predict_df <- tidyr::expand_grid(
  n = 0:N,  # 試行回数
  x = x_vec # 確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    mu_star     = trace_mu_star_i[n+1],     # 平均パラメータ
    lambda_star = trace_lambda_star_i[n+1], # 精度パラメータ
    sigma_star  = 1/sqrt(lambda_star),      # 標準偏差パラメータ
    dens        = dnorm(x = x, mean = mu_star, sd = sigma_star) # 確率密度
  )


### 推移の作図 -----

#### 事後分布の作図 -----

# 事後分布のラベルを作成
anim_param_df <- tibble::tibble(
  n         = 0:N, 
  m         = trace_m_i, 
  lambda_mu = trace_lambda_mu_i
) |> 
  dplyr::mutate(
    posterior_param_lbl = sprintf(
      fmt = "list(N == '%s', mu[truth] == %s, hat(m) == '%s', hat(lambda)[mu] == '%s')", 
      formatC(n,         digits = 0, format = "d"), 
      formatC(mu_truth,  digits = 2, format = "f"), 
      formatC(m,         digits = 2, format = "f"), 
      formatC(lambda_mu, digits = 5, format = "f")
    )
  )

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_vline(
    mapping = aes(xintercept = mu_truth, color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_line(
    data    = anim_posterior_df, 
    mapping = aes(x = mu, y = dens, color = "posterior"), 
    linewidth = 1
  ) + # 事後分布
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
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", "purple", "hotpink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
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
    title = "Gaussian distribution", 
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
    file = "figure/gaussian_model/parameter_updates_mean/posterior.mp4"
  )
)


#### 予測分布の作図 -----

# 予測分布のラベルを作成
anim_param_df <- tibble::tibble(
  n           = 0:N, 
  mu_star     = trace_mu_star_i, 
  lambda_star = trace_lambda_star_i
) |> 
  dplyr::mutate(
    predict_param_lbl = sprintf(
      fmt = "list(N == '%s', mu[truth] == %s, lambda == %s, hat(mu)['*'] == '%s', hat(lambda)['*'] == '%s')", 
      formatC(n,           digits = 0, format = "d"), 
      formatC(mu_truth,    digits = 2, format = "f"), 
      formatC(lambda,      digits = 5, format = "f"), 
      formatC(mu_star,     digits = 2, format = "f"), 
      formatC(lambda_star, digits = 5, format = "f")
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
    title = "Gaussian distribution", 
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
    file = "figure/gaussian_model/parameter_updates_mean/predict.mp4"
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


# p(μ)軸の範囲を設定
u <- 0.05
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
  m           <- trace_m_i[i]           # 平均パラメータ
  lambda_mu   <- trace_lambda_mu_i[i]   # 精度パラメータ
  mu_star     <- trace_mu_star_i[i]     # 平均パラメータ
  lambda_star <- trace_lambda_star_i[i] # 精度パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x = x # 観測値
  )
  data_df <- tibble::tibble(
    x = x_n[0:n] # 観測値
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
    fmt = "list(N == '%s', mu[truth] == %s, lambda == %s, paste(E(x) == mu[truth], {} == '%s'), bar(x) == '%s')", 
    formatC(n,        digits = 0, format = "d"), 
    formatC(mu_truth, digits = 2, format = "f"), 
    formatC(lambda,   digits = 5, format = "f"), 
    formatC(E_x,      digits = 2, format = "f"), 
    formatC(bar_x,    digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 軸目盛を設定:(目盛の共通化用)
  dens_vals <- dummy_built$layout$panel_params[[1]]$y$breaks            # 確率密度軸目盛を取得
  freq_vals <- dens_vals * ifelse(test = n>0, yes = bin_size*n, no = 1) # 度数軸目盛に変換
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth), 
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
        breaks    = c(E_x, bar_x), 
        labels    = c(expression(E(x)), expression(bar(x)))
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
  E_mu <- m
  
  # 事後分布の確率密度を計算
  posterior_df <- tibble::tibble(
    mu   = mu_vec, # 確率変数
    dens = dnorm(x = mu, mean = m, sd = 1/sqrt(lambda_mu)) # 確率密度
  )
  
  # 事後分布のラベルを作成
  posterior_param_lbl <- sprintf(
    fmt = "list(hat(m) == '%s', hat(lambda)[mu] == '%s', paste(E(mu) == hat(m), {} == '%s'))", 
    formatC(m,         digits = 2, format = "f"), 
    formatC(lambda_mu, digits = 5, format = "f"), 
    formatC(E_mu,      digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth, color = "model"), 
      linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_mu), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_line(
      data    = posterior_df, 
      mapping = aes(x = mu, y = dens, color = "posterior"), 
      linewidth = 1
    ) + # 事後分布
    geom_point(
      data    = data_df, 
      mapping = aes(x = x, y = 0), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x, y = 0), 
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
      xlim = c(mu_min, mu_max),       # (目盛の共通化用)
      ylim = c(0, posterior_dens_max) # 表示範囲を固定
    ) + 
    labs(
      title = "Gaussian distribution", 
      subtitle = posterior_param_lbl, 
      x = expression(mu), 
      y = "density"
    )
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x <- mu_star
  
  # 予測分布の確率密度を計算
  predict_df <- tibble::tibble(
    x    = x_vec, # 確率変数
    dens = dnorm(x = x, mean = mu_star, sd = 1/sqrt(lambda_star)) # 確率密度
  )
  
  # 予測分布のラベルを作成
  predict_param_lbl <- sprintf(
    fmt = "list(hat(mu)['*'] == '%s', hat(lambda)['*'] == '%s', paste(E(x) == hat(mu)['*'], {} == '%s'))", 
    formatC(mu_star,     digits = 2, format = "f"), 
    formatC(lambda_star, digits = 5, format = "f"), 
    formatC(E_x,         digits = 2, format = "f")
  ) |> 
    parse(text = _)
  
  # 予測分布を作図
  predict_graph <- ggplot() + 
    geom_vline(
      mapping = aes(xintercept = mu_truth), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_x), 
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
      title = "Gaussian distribution", 
      subtitle = predict_param_lbl, 
      x = expression(x), 
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
    model_graph, posterior_graph, predict_graph, 
    nrow = 3, ncol = 1, 
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
    width = 9, height = 12, units = "in", dpi = 100
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
    path = "figure/gaussian_model/parameter_updates_mean/observation.mp4", 
    framerate = 10
  ) -> tmp_path # mp4ファイルを書出


