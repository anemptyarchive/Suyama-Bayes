
# 多次元ガウスモデル ------------------------------------------------------------

# chapter 3.4.1
# 平均が未知の場合
# ベイズ推論
# 学習推移の可視化


# パッケージの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の可視化 -----------------------------------------------------------

### 生成分布(ガウス分布)の設定 -----

# 次元数を指定
D <- 2 # (固定)

# 真のパラメータを指定
mu_truth_d <- c(25, 50)

# 既知のパラメータを指定
sigma2_dd <- c(
  900, -100, 
  -100, 400
) |> 
  matrix(nrow = D, ncol = D)

# 精度行列に変換
lambda_dd <- solve(sigma2_dd) # (処理の効率化用)


### 観測データの設定 -----

# シードを設定:(ノートとの対応用)
set.seed(86)

# データ数(試行回数)を指定
N <- 100

# 観測データを生成
x_nd <- mvnfast::rmvn(n = N, mu = mu_truth_d, sigma = sigma2_dd)


### 変数の設定 -----

# x軸の範囲を設定
k <- 4
u <- 5
x_1_size <- sqrt(sigma2_dd[1, 1]) |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., abs(x_nd[, 1]-mu_truth_d[1]))})() |> # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_2_size <- sqrt(sigma2_dd[2, 2]) |> # 標準偏差
  (\(.) {. * k})() |> # 定数倍
  #(\(.) {max(., abs(x_nd[, 2]-mu_truth_d[2]))})() |> # サンプルと比較
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
x_1_min <- mu_truth_d[1] - x_1_size
x_1_max <- mu_truth_d[1] + x_1_size
x_2_min <- mu_truth_d[2] - x_2_size
x_2_max <- mu_truth_d[2] + x_2_size
cat("x1 size:", x_1_min, x_1_max)
cat("x2 size:", x_2_min, x_2_max)

# x軸の値を作成
x_1_vec <- seq(from = x_1_min, to = x_1_max, length.out = 101)
x_2_vec <- seq(from = x_2_min, to = x_2_max, length.out = 101)


# μ軸の範囲を設定
mu_1_min <- x_1_min # (固定)
mu_1_max <- x_1_max # (固定)
mu_2_min <- x_2_min # (固定)
mu_2_max <- x_2_max # (固定)
cat("μ1 size:", mu_1_min, mu_1_max)
cat("μ2 size:", mu_2_min, mu_2_max)

# μ軸の値を作成
mu_1_vec <- seq(from = mu_1_min, to = mu_1_max, length.out = 101)
mu_2_vec <- seq(from = mu_2_min, to = mu_2_max, length.out = 101)


### パラメータの更新 -----

#### 逐次更新の場合 -----

# 事前分布のパラメータを初期化
m_d          <- rep(0, times = D)
sigma2_mu_dd <- diag(D) * 100^2
lambda_mu_dd <- solve(sigma2_mu_dd)

# 予測分布のパラメータを計算:式(3.109, 3.110)
mu_star_d      <- m_d
lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))

# 受け皿を初期化
trace_m_di            <- matrix(NA, nrow = D, ncol = N+1)
trace_lambda_mu_ddi   <- array(NA, dim = c(D, D, N+1))
trace_mu_star_di      <- matrix(NA, nrow = D, ncol = N+1)
trace_lambda_star_ddi <- array(NA, dim = c(D, D, N+1))

# 初期値を記録
trace_m_di[, 1]              <- m_d
trace_lambda_mu_ddi[, , 1]   <- lambda_mu_dd
trace_mu_star_di[, 1]        <- mu_star_d
trace_lambda_star_ddi[, , 1] <- lambda_star_dd

# ベイズ推論による更新
for(n in 1:N){
  
  # 観測データを取得
  x_d <- x_nd[n, ]
  
  # 事後分布のパラメータを更新:式(3.102, 3.103)
  old_lambda_mu_dd <- lambda_mu_dd
  lambda_mu_dd     <- lambda_mu_dd + lambda_dd
  m_d              <- (solve(lambda_mu_dd) %*% (lambda_dd %*% x_d + old_lambda_mu_dd %*% m_d)) |> 
    as.vector()
  
  # 予測分布のパラメータを更新:式(3.109, 3.110)
  mu_star_d      <- m_d
  lambda_star_dd <- solve(solve(lambda_dd) + solve(lambda_mu_dd))
  
  # 更新値を記録
  trace_m_di[, n+1]              <- m_d
  trace_lambda_mu_ddi[, , n+1]   <- lambda_mu_dd
  trace_mu_star_di[, n+1]        <- mu_star_d
  trace_lambda_star_ddi[, , n+1] <- lambda_star_dd
  
  # 動作確認
  message("\r", n, " / ", N, appendLF = FALSE)
}


#### 一括更新の場合 -----

# 事前分布のパラメータを初期化
m_d          <- rep(0, times = D)
sigma2_mu_dd <- diag(D) * 100^2
lambda_mu_dd <- solve(sigma2_mu_dd)

# 事後分布のパラメータを計算:式(3.102, 3.103)
trace_lambda_mu_ddi <- rep(0:N, each = D*D) * array(lambda_dd, dim = c(D, D, N+1)) + array(lambda_mu_dd, dim = c(D, D, N+1))
trace_m_di <- lapply(
  X   = 0:N, 
  FUN = \(n) {
    (solve(trace_lambda_mu_ddi[, , n+1]) %*% (lambda_dd %*% colSums(x_nd[0:n, , drop = FALSE]) + lambda_mu_dd %*% m_d)) |> 
      as.vector()
  }
) |> 
  unlist() |> 
  matrix(nrow = D, ncol = N+1)

# 予測分布のパラメータを計算:式(3.109', 3.110')
trace_mu_star_di <- trace_m_di
tmp_trace_lambda_star_ddi <- lapply(
  X   = 0:N, 
  FUN = \(n) {
    solve(solve(lambda_dd) + solve(trace_lambda_mu_ddi[, , n+1]))
  }
) |> 
  unlist() |> 
  array(dim = c(D, D, N+1))


### 分布の計算 -----

# サンプルデータを格納
anim_sample_df <- tibble::tibble(
  n   = 0:N,              # データ番号
  x_1 = c(NA, x_nd[, 1]), # 1軸の観測値
  x_2 = c(NA, x_nd[, 2])  # 2軸の観測値
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
    x_1 = x_nd[n, 1], # 1軸の観測値
    x_2 = x_nd[n, 2]  # 2軸の観測値
  ) |> 
  tidyr::complete(
    i = 0:N, 
    fill = list(n = NA, x_1 = NA, x_2 = NA)
  ) |> # 初期値用のデータを補完
  dplyr::select(n = i, x_1, x_2)


# 生成分布の確率密度を計算
model_df <- tidyr::expand_grid(
  x_1 = x_1_vec, # 1軸の確率変数
  x_2 = x_2_vec  # 2軸の確率変数
) |> # 格子点を作成
  dplyr::mutate(
    mu     = list(mu_truth_d),         # 平均パラメータ
    lambda = list(lambda_dd),          # 精度パラメータ
    sigma2 = list(solve(lambda[[1]])), # 分散パラメータ
    dens   = mvnfast::dmvn(X = cbind(x_1, x_2), mu = mu[[1]], sigma = sigma2[[1]]) # 確率密度
  )


# 事後分布の確率密度を計算
anim_posterior_df <- tidyr::expand_grid(
  n    = 0:N,      # データ番号
  mu_1 = mu_1_vec, # 1軸の確率変数
  mu_2 = mu_2_vec  # 2軸の確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    m         = list(trace_m_di[, unique(n+1)]),            # 平均パラメータ
    lambda_mu = list(trace_lambda_mu_ddi[, , unique(n+1)]), # 精度パラメータ
    sigma2_mu = list(solve(lambda_mu[[1]])),                # 分散パラメータ
    dens      = mvnfast::dmvn(X = cbind(mu_1, mu_2), mu = m[[1]], sigma = sigma2_mu[[1]]), # 確率密度
    .by = n
  )

# 予測分布の確率密度を計算
anim_predict_df <- tidyr::expand_grid(
  n   = 0:N,     # データ番号
  x_1 = x_1_vec, # 1軸の確率変数
  x_2 = x_2_vec  # 2軸の確率変数
) |> # 試行ごとに変数を複製
  dplyr::mutate(
    mu_star     = list(trace_mu_star_di[, unique(n+1)]),        # 平均パラメータ
    lambda_star = list(trace_lambda_star_ddi[, , unique(n+1)]), # 精度パラメータ
    sigma2_star = list(solve(lambda_star[[1]])),                # 分散パラメータ
    dens      = mvnfast::dmvn(X = cbind(x_1, x_2), mu = mu_star[[1]], sigma = sigma2_star[[1]]), # 確率密度
    .by = n
  )


### 推移の作図 -----

#### 事後分布の作図 -----

# 事後分布のラベルを作成
anim_param_df <- tibble::tibble(
  n = 0:N
) |> 
  dplyr::mutate(
    m         = list(trace_m_di[, n+1]), 
    lambda_mu = list(trace_lambda_mu_ddi[, , n+1]), 
    .by = n
  ) |> 
  dplyr::mutate(
    posterior_param_lbl = sprintf(
      fmt = paste0(
        "list(", 
        "N == '%s', ", 
        "mu[truth] == bgroup('(', atop('%s'), ')'), ", 
        "hat(m) == bgroup('(', atop('%s'), ')'), ", 
        "hat(Lambda)[mu] == bgroup('(', atop(list('%s'), list('%s')), ')')", 
        ")"
      ), 
      formatC(n,                   digits = 0, format = "d"), 
      formatC(mu_truth_d,          digits = 2, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(m[[1]],              digits = 2, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_mu[[1]][1, ], digits = 5, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_mu[[1]][2, ], digits = 5, format = "f") |> 
        paste0(collapse = "', '")
    ), 
    .by = n
  )

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_contour_filled(
    data    = anim_posterior_df, 
    mapping = aes(x = mu_1, y = mu_2, z = dens, fill = after_stat(level), color = "posterior"), 
    alpha = 0.6
  ) + # 事後分布
  geom_vline(
    mapping = aes(xintercept = mu_truth_d[1], color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_hline(
    mapping = aes(yintercept = mu_truth_d[2], color = "model"), 
    linewidth = 1, linetype = "dashed"
  ) + # 真のパラメータ
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x_1, y = x_2), 
    na.rm = TRUE, 
    color = "hotpink", alpha = 0.33, size = 2.5
  ) + # 観測データ
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x_1, y = x_2, color = "sample"), 
    na.rm = TRUE, 
    size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = posterior_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -1
  ) + # パラメータラベル
  geom_text(
    mapping = aes(x = 0.5*(mu_1_min+mu_1_max), y = -Inf, label = "mu[1]"), 
    parse = TRUE, hjust = 0.5, vjust = 2.5, 
    size = 4
  ) + # 変数ラベル
  geom_text(
    mapping = aes(x = -Inf, y = 0.5*(mu_2_min+mu_2_max), label = "mu[2]"), 
    parse = TRUE, angle = 90, hjust = 0.5, vjust = -2.5, 
    size = 4
  ) + # 変数ラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_x_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = mu_truth_d[1], 
      labels    = expression(mu[1]^{truth})
    ) # パラメータラベル
  ) + 
  scale_y_continuous(
    sec.axis = sec_axis(
      transform = ~ ., 
      breaks    = mu_truth_d[2], 
      labels    = expression(mu[2]^{truth})
    ) # パラメータラベル
  ) + 
  scale_color_manual(
    breaks = c("model", "posterior", "sample"), 
    values = c("red", NA, "hotpink"), 
    labels = c("true parameter", "posterior distribution", "observation data"), 
    name = ""
  ) + # (凡例の表示用)
  guides(
    color = guide_legend(override.aes = list(linewidth = 0.5), order = 1), 
    fill  = guide_legend(order = 2)
  ) + 
  theme(
    axis.title.x  = element_text(margin = margin(t = 20)), # (変数ラベル用の空行サイズ)
    axis.title.y  = element_text(margin = margin(r = 20)), # (変数ラベル用の空行サイズ)
    plot.subtitle = element_text(margin = margin(b = 40))  # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    xlim = c(mu_1_min, mu_1_max), # (変数ラベルの対応用)
    ylim = c(mu_2_min, mu_2_max), # (変数ラベルの対応用)
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "multivariate Gaussian distribution", 
    subtitle = NULL, # (パラメータラベル用の空行)
    fill = "density", 
    x = NULL, # (変数ラベル用の空行)
    y = NULL  # (変数ラベル用の空行)
  )

# 動画を作成
gganimate::animate(
  plot = posterior_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(
    file = "figure/multivariate_gaussian_model/parameter_updates_mean/posterior.mp4"
  )
)


#### 予測分布の作図 -----

# 予測分布のラベルを作成
anim_param_df <- tibble::tibble(
  n = 0:N
) |> 
  dplyr::mutate(
    mu_star     = list(trace_mu_star_di[, n+1]), 
    lambda_star = list(trace_lambda_star_ddi[, , n+1]), 
    .by = n
  ) |> 
  dplyr::mutate(
    predict_param_lbl = sprintf(
      fmt = paste0(
        "list(", 
        "N == '%s', ", 
        "mu[truth] == bgroup('(', atop('%s'), ')'), ", 
        "Lambda == bgroup('(', atop(list('%s'), list('%s')), ')'), ", 
        "hat(mu)['*'] == bgroup('(', atop('%s'), ')'), ", 
        "hat(Lambda)['*'] == bgroup('(', atop(list('%s'), list('%s')), ')')", 
        ")"
      ), 
      formatC(n,                     digits = 0, format = "d"), 
      formatC(mu_truth_d,            digits = 2, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_dd[1, ],        digits = 5, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_dd[2, ],        digits = 5, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(mu_star[[1]],          digits = 2, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_star[[1]][1, ], digits = 5, format = "f") |> 
        paste0(collapse = "', '"), 
      formatC(lambda_star[[1]][2, ], digits = 5, format = "f") |> 
        paste0(collapse = "', '")
    ), 
    .by = n
  )

# p(x)軸の範囲を設定
u <- 0.00005
dens_min <- 0
dens_max <- anim_predict_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ

# 等高線の目盛を設定
dummy_graph <- ggplot() + # 目盛の設定用のダミー
  geom_contour(
    data    = model_df, 
    mapping = aes(x = x_1, y = x_2, z = dens)
  ) # 真の分布
dummy_built     <- ggplot_build(dummy_graph)      # 図の情報を取得
dummy_dens_vals <- dummy_built$data[[1]]$level |> # 目盛の情報を取得
  unique() |> 
  sort()
predict_dens_vals <- c(
  ifelse(test = dens_min < min(dummy_dens_vals), yes = dens_min, no = NULL), 
  dummy_dens_vals, 
  ifelse(test = dens_max > max(dummy_dens_vals), yes = dens_max, no = NULL)
) # (塗りつぶしとの対応用)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_contour_filled(
    data    = anim_predict_df, 
    mapping = aes(x = x_1, y = x_2, z = dens, fill = after_stat(level), alpha = "predict"), 
    breaks = predict_dens_vals # (目盛の共通化用)
  ) + # 予測分布
  geom_contour(
    data    = model_df, 
    mapping = aes(x = x_1, y = x_2, z = dens, color = after_stat(level), alpha = "model"), 
    breaks = predict_dens_vals, # (目盛の共通化用)
    linewidth = 1, linetype = "dashed"
  ) + # 真の分布
  geom_point(
    data    = anim_data_df, 
    mapping = aes(x = x_1, y = x_2), 
    na.rm = TRUE, 
    color = "hotpink", alpha = 0.33, size = 2.5
  ) + # 観測データ
  geom_point(
    data    = anim_sample_df, 
    mapping = aes(x = x_1, y = x_2, alpha = "sample"), 
    na.rm = TRUE, 
    color = "hotpink", size = 5
  ) + # 観測データ
  geom_text(
    data    = anim_param_df, 
    mapping = aes(x = -Inf, y = Inf, label = predict_param_lbl), 
    parse = TRUE, hjust = 0, vjust = -0.3
  ) + # パラメータラベル
  geom_text(
    mapping = aes(x = 0.5*(x_1_min+x_1_max), y = -Inf, label = "x[1]"), 
    parse = TRUE, hjust = 0.5, vjust = 2.5, 
    size = 4
  ) + # 変数ラベル
  geom_text(
    mapping = aes(x = -Inf, y = 0.5*(x_2_min+x_2_max), label = "x[2]"), 
    parse = TRUE, angle = 90, hjust = 0.5, vjust = -2.5, 
    size = 4
  ) + # 変数ラベル
  gganimate::transition_manual(frames = n) + # フレーム制御
  scale_color_viridis_c(
    option = "viridis"
  ) + # (配色の共通化用)
  scale_alpha_manual(
    breaks = c("model", "predict", "sample"), 
    values = c(1, 0.6, 1), 
    labels = c("true model", "predict distribution", "observation data"), 
    name = ""
  ) + # (凡例の表示用)
  guides(
    alpha = guide_legend(
      override.aes = list(
        linewidth = c(0.5, 0.5, NA),
        linetype  = c("dashed", NA, NA),
        shape     = c(NA, NA, 19)
      ), 
      order = 1
    ), 
    fill  = guide_legend(order = 2), 
    color = "none"
  ) + 
  theme(
    axis.title.x  = element_text(margin = margin(t = 15)), # (変数ラベル用の空行サイズ)
    axis.title.y  = element_text(margin = margin(r = 15)), # (変数ラベル用の空行サイズ)
    plot.subtitle = element_text(margin = margin(b = 40))  # (パラメータラベル用の空行サイズ)
  ) + 
  coord_cartesian(
    xlim = c(x_1_min, x_1_max), # (変数ラベルの対応用)
    ylim = c(x_2_min, x_2_max), # (変数ラベルの対応用)
    clip = "off" # (パラメータラベル用の枠外表示設定)
  ) + 
  labs(
    title = "multivariate Gaussian distribution", 
    subtitle = NULL, # (パラメータラベル用の空行)
    fill = "density", 
    x = NULL, # (変数ラベル用の空行)
    y = NULL  # (変数ラベル用の空行)
  )

# 動画を作成
gganimate::animate(
  plot = predict_graph, 
  nframes = N+1, fps = 10, 
  width = 9, height = 6, units = "in", res = 100, 
  renderer = gganimate::av_renderer(
    file = "figure/multivariate_gaussian_model/parameter_updates_mean/predict.mp4"
  )
)


#### 観測データと分布の関係 -----

# 一時書き出し先を指定
dir_path <- "figure/tmp_folder"


# 階級数を指定
bin_num <- 40

# 階級幅を計算
bin_1_size <- (x_1_max - x_1_min) / bin_num # 1軸の階級幅
bin_2_size <- (x_2_max - x_2_min) / bin_num # 2軸の階級幅
bin_1_min <- x_1_min - 0.5*bin_1_size # 1軸の境界値の最小値
bin_1_max <- x_1_min + 0.5*bin_1_size # 1軸の境界値の最大値
bin_2_min <- x_2_min - 0.5*bin_2_size # 2軸の境界値の最小値
bin_2_max <- x_2_min + 0.5*bin_2_size # 2軸の境界値の最大値
center_1_vec <- seq(from = x_1_min, to = x_1_max, by = bin_1_size) # 1軸の階級値
center_2_vec <- seq(from = x_2_min, to = x_2_max, by = bin_2_size) # 2軸の階級値


# p(μ)軸の範囲を設定
u <- 0.005
posterior_dens_max <- anim_posterior_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
posterior_dens_max

# 等高線の目盛を設定
level_num <- 31
posterior_dens_vals <- seq(from = 0, to = posterior_dens_max, length.out = level_num)
posterior_dens_vals

# p(x)軸の範囲を設定
u <- 0.00005
predict_dens_min <- 0
predict_dens_max <- anim_predict_df |> 
  dplyr::pull(dens) |> 
  max() |> 
  (\(.) {ceiling(. /u)*u})() # u単位で切り上げ
predict_dens_max

# 等高線の目盛を設定
dummy_graph <- ggplot() + # 目盛の設定用のダミー
  geom_contour(
    data    = model_df, 
    mapping = aes(x = x_1, y = x_2, z = dens)
  ) # 真の分布
dummy_built     <- ggplot_build(dummy_graph)      # 図の情報を取得
dummy_dens_vals <- dummy_built$data[[1]]$level |> # 目盛の情報を取得
  unique() |> 
  sort()
predict_dens_vals <- c(
  ifelse(test = predict_dens_min < min(dummy_dens_vals), yes = predict_dens_min, no = NULL), 
  dummy_dens_vals, 
  ifelse(test = predict_dens_max > max(dummy_dens_vals), yes = predict_dens_max, no = NULL)
) # (塗りつぶしとの対応用)
predict_dens_vals

# 等高線の配色を設定
color_name <- "YlOrRd" # カラーマップを指定
color_num  <- 9        # カラーマップの色数を設定
level_num  <- length(predict_dens_vals)
color_vals <- colorRampPalette(colors = RColorBrewer::brewer.pal(n = color_num, name = color_name))(n = level_num-1) |> # 色数を拡張
  rev()


# 試行ごとに作図
for(n in 0:N) {
  
  ##### パラメータの取得 -----
  
  # 値を取得
  i              <- n + 1     # 試行インデックス
  x_d            <- x_nd[n, ] # 観測値
  m_d            <- trace_m_di[, i]              # 平均パラメータ
  lambda_mu_dd   <- trace_lambda_mu_ddi[, , i]   # 精度パラメータ
  mu_star_d      <- trace_mu_star_di[, i]        # 平均パラメータ
  lambda_star_dd <- trace_lambda_star_ddi[, , i] # 精度パラメータ
  
  # 観測データを格納
  sample_df <- tibble::tibble(
    x_1 = x_d[1], # 1軸の観測値
    x_2 = x_d[2]  # 2軸の観測値
  )
  data_df <- tibble::tibble(
    x_1 = x_nd[0:n, 1], # 1軸の観測値
    x_2 = x_nd[0:n, 2]  # 2軸の観測値
  )
  
  ##### 観測データの作図 -----
  
  # 生成分布の期待値を計算
  E_x_d <- mu_truth_d
  
  # 観測データの標本平均を計算
  bar_x_d <- apply(X = x_nd[0:n, , drop = FALSE], MARGIN = 2, FUN = mean)
  
  # 観測データを集計
  obs_df <- data_df |> 
    dplyr::mutate(
      bin_1_i  = (x_1 - bin_1_min) %/% bin_1_size,         # 1軸の階級番号
      bin_2_i  = (x_2 - bin_2_min) %/% bin_2_size,         # 2軸の階級番号
      center_1 = bin_1_min + (bin_1_i + 0.5) * bin_1_size, # 1軸の階級値
      center_2 = bin_2_min + (bin_2_i + 0.5) * bin_2_size  # 2軸の階級値
    ) |> 
    dplyr::count(
      center_1, center_2, name = "freq" # 度数
    ) |> 
    dplyr::mutate(
      dens = freq / (bin_1_size * bin_2_size * N) # 密度
    ) |> 
    tidyr::complete(
      center_1 = center_1_vec, 
      center_2 = center_2_vec, 
      fill = list(freq = 0, dens = 0)
    ) # 未観測値を補完
  
  # 生成分布のラベルを作成
  model_param_lbl <- sprintf(
    fmt = paste0(
      "list(", 
      #"N == '%s', ", 
      "mu[truth] == bgroup('(', atop('%s'), ')'), ", 
      "Lambda == bgroup('(', atop(list('%s'), list('%s')), ')'), ", 
      "paste(E(x) == mu[truth], {} == bgroup('(', atop('%s'), ')')), ", 
      "bar(x) == bgroup('(', atop('%s'), ')')", 
      ")"
    ), 
    #formatC(n,              digits = 0, format = "d"), 
    formatC(mu_truth_d,     digits = 2, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_dd[1, ], digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_dd[2, ], digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(E_x_d,          digits = 2, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(bar_x_d,        digits = 2, format = "f") |> 
      paste0(collapse = "', '")
  ) |> 
    parse(text = _)
  
  # 観測データを作図
  model_graph <- ggplot() + 
    geom_tile(
      data    = obs_df, 
      mapping = aes(x = center_1, y = center_2, fill = dens, linetype = "sample"), 
      alpha = 0.5
    ) + # 観測データ
    geom_contour(
      data    = model_df, 
      mapping = aes(x = x_1, y = x_2, z = dens, color = after_stat(level), linetype = "model"), 
      breaks = predict_dens_vals, # (目盛の共通化用)
      linewidth = 1
    ) + # 生成分布
    geom_vline(
      mapping = aes(xintercept = E_x_d[1]), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_hline(
      mapping = aes(yintercept = E_x_d[2]), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_vline(
      mapping = aes(xintercept = bar_x_d[1]), 
      color = "hotpink", linewidth = 1, linetype = "dashed"
    ) + # 標本平均
    geom_hline(
      mapping = aes(yintercept = bar_x_d[2]), 
      color = "hotpink", linewidth = 1, linetype = "dashed"
    ) + # 標本平均
    geom_point(
      data    = data_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(E_x_d[1], bar_x_d[1]), 
        labels    = c(expression(E(x[1])), expression(bar(x[1])))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(E_x_d[2], bar_x_d[2]), 
        labels    = c(expression(E(x[2])), expression(bar(x[2])))
      ) # パラメータラベル
    ) + 
    scale_fill_gradientn(
      colors = color_vals, 
      limits = c(predict_dens_min, predict_dens_max) # (目盛の共通化用)
    ) + 
    scale_colour_distiller(
      palette = color_name, direction = -1, 
      limits = c(predict_dens_min, predict_dens_max) # (目盛の共通化用)
    ) + 
    scale_linetype_manual(
      breaks = c("model", "sample"), 
      values = c("dashed", "blank"), 
      labels = c("true model", "observation data"), 
      name = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      linetype = guide_legend(override.aes = list(color = "red", linewidth = 0.5)), 
      fill     = "none", 
      color    = "none"
    ) + 
    coord_cartesian(
      xlim = c(x_1_min, x_1_max), # (目盛の共通化用)
      ylim = c(x_2_min, x_2_max), # (目盛の共通化用)
      ratio = 1
    ) + 
    labs(
      title = "multivariate Gaussian distribution", 
      subtitle = model_param_lbl, 
      fill  = "density", 
      color = "density", 
      x = expression(x[1]), 
      y = expression(x[2])
    )
  
  ##### 事後分布の作図 -----
  
  # 事後分布の期待値を計算
  E_mu_d <- m_d
  
  # 事後分布の確率密度を計算
  posterior_df <- tidyr::expand_grid(
    mu_1 = mu_1_vec, # 1軸の確率変数
    mu_2 = mu_2_vec  # 2軸の確率変数
  ) |> # 格子点を作成
    dplyr::mutate(
      m         = list(m_d),                   # 平均パラメータ
      lambda_mu = list(lambda_mu_dd),          # 精度パラメータ
      sigma2_mu = list(solve(lambda_mu[[1]])), # 分散パラメータ
      dens      = mvnfast::dmvn(X = cbind(mu_1, mu_2), mu = m[[1]], sigma = sigma2_mu[[1]]) # 確率密度
    )
  
  # 事後分布のラベルを作成
  posterior_param_lbl <- sprintf(
    fmt = paste0(
      "list(", 
      "hat(m) == bgroup('(', atop('%s'), ')'), ", 
      "hat(Lambda)[mu] == bgroup('(', atop(list('%s'), list('%s')), ')'), ", 
      "paste(E(mu) == hat(m), {} == bgroup('(', atop('%s'), ')'))", 
      ")"
    ), 
    formatC(m_d,                digits = 2, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_mu_dd[1, ],  digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_mu_dd[2, ],  digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(E_mu_d,             digits = 2, format = "f") |> 
      paste0(collapse = "', '")
  ) |> 
    parse(text = _)
  
  # 事後分布を作図
  posterior_graph <- ggplot() + 
    geom_contour_filled(
      data    = posterior_df, 
      mapping = aes(x = mu_1, y = mu_2, z = dens, fill = after_stat(level), linetype = "posterior"), 
      breaks = posterior_dens_vals, # (目盛の共通化用)
      alpha = 0.6
    ) + # 事後分布
    geom_vline(
      mapping = aes(xintercept = mu_truth_d[1], linetype = "model"), 
      color = "red", linewidth = 1
    ) + # 真のパラメータ
    geom_hline(
      mapping = aes(yintercept = mu_truth_d[2], linetype = "model"), 
      color = "red", linewidth = 1
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_mu_d[1]), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_hline(
      mapping = aes(yintercept = E_mu_d[2]), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_point(
      data    = data_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth_d[1], E_mu_d[1]), 
        labels    = c(expression(mu[1]^{truth}), expression(E(mu[1])))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth_d[2], E_mu_d[2]), 
        labels    = c(expression(mu[2]^{truth}), expression(E(mu[2])))
      ) # パラメータラベル
    ) + 
    scale_fill_viridis_d(
      drop = FALSE # (目盛の共通化用)
    ) + 
    scale_linetype_manual(
      breaks = c("model", "posterior"), 
      values = c("dashed", "blank"), 
      labels = c("true parameter", "posterior distribution"), 
      name = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      linetype = guide_legend(override.aes = list(linewidth = 0.5)), 
      fill     = "none"
    ) + 
    coord_cartesian(
      xlim = c(mu_1_min, mu_1_max), # (目盛の共通化用)
      ylim = c(mu_2_min, mu_2_max), # (目盛の共通化用)
      ratio = 1
    ) + 
    labs(
      title = "multivariate Gaussian distribution", 
      subtitle = posterior_param_lbl, 
      fill = "density", 
      x = expression(mu[1]), 
      y = expression(mu[2])
    )
  
  ##### 予測分布の作図 -----
  
  # 予測分布の期待値を計算
  E_x_star_d <- mu_star_d
  
  # 予測分布の確率密度を計算
  predict_df <- tidyr::expand_grid(
    x_1 = x_1_vec, # 1軸の確率変数
    x_2 = x_2_vec  # 2軸の確率変数
  ) |> # 格子点を作成
    dplyr::mutate(
      mu_star     = list(mu_star_d),               # 平均パラメータ
      lambda_star = list(lambda_star_dd),          # 精度パラメータ
      sigma2_star = list(solve(lambda_star[[1]])), # 分散パラメータ
      dens = mvnfast::dmvn(X = cbind(x_1, x_2), mu = mu_star[[1]], sigma = sigma2_star[[1]]) # 確率密度
    )
  
  # 予測分布のラベルを作成
  predict_param_lbl <- sprintf(
    fmt = paste0(
      "list(", 
      "hat(mu)['*'] == bgroup('(', atop('%s'), ')'), ", 
      "hat(Lambda)['*'] == bgroup('(', atop(list('%s'), list('%s')), ')'), ", 
      "paste(E(x['*']) == hat(mu)['*'], {} == bgroup('(', atop('%s'), ')'))", 
      ")"
    ), 
    formatC(mu_star_d,           digits = 2, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_star_dd[1, ], digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(lambda_star_dd[2, ], digits = 5, format = "f") |> 
      paste0(collapse = "', '"), 
    formatC(E_x_star_d,          digits = 2, format = "f") |> 
      paste0(collapse = "', '")
  ) |> 
    parse(text = _)
  
  # 予測分布を作図
  predict_graph <- ggplot() + 
    geom_contour_filled(
      data    = predict_df, 
      mapping = aes(x = x_1, y = x_2, z = dens, fill = after_stat(level), linetype = "predict"), 
      breaks = predict_dens_vals, # (目盛の共通化用)
      alpha = 0.6
    ) + # 予測分布
    geom_contour(
      data    = model_df, 
      mapping = aes(x = x_1, y = x_2, z = dens, color = after_stat(level), linetype = "model"), 
      breaks = predict_dens_vals, # (目盛の共通化用)
      linewidth = 1
    ) + # 真の分布
    geom_vline(
      mapping = aes(xintercept = mu_truth_d[1]), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_hline(
      mapping = aes(yintercept = mu_truth_d[2]), 
      color = "red", linewidth = 1, linetype = "dashed"
    ) + # 真のパラメータ
    geom_vline(
      mapping = aes(xintercept = E_x_star_d[1]), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_hline(
      mapping = aes(yintercept = E_x_star_d[2]), 
      color = "purple", linewidth = 1, linetype = "dashed"
    ) + # 期待値
    geom_point(
      data    = data_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", alpha = 0.33, size = 2.5
    ) + # 観測データ
    geom_point(
      data    = sample_df, 
      mapping = aes(x = x_1, y = x_2), 
      na.rm = TRUE, 
      color = "hotpink", size = 5
    ) + # 観測データ
    scale_x_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth_d[1], E_x_star_d[1]), 
        labels    = c(expression(mu[1]^{truth}), expression(E(x[1]^{'*'})))
      ) # パラメータラベル
    ) + 
    scale_y_continuous(
      sec.axis = sec_axis(
        transform = ~ ., 
        breaks    = c(mu_truth_d[2], E_x_star_d[2]), 
        labels    = c(expression(mu[2]^{truth}), expression(E(x[2]^{'*'})))
      ) # パラメータラベル
    ) + 
    scale_fill_viridis_d(
      drop = FALSE # (目盛の共通化用)
    ) + 
    scale_color_distiller(
      palette = color_name, direction = -1, 
      limits = c(predict_dens_min, predict_dens_max) # (目盛の共通化用)
    ) + 
    scale_linetype_manual(
      breaks = c("model", "predict"), 
      values = c("dashed", "blank"), 
      labels = c("true model", "predict distribution"), 
      name = ""
    ) + # (凡例の表示用)
    theme(
      legend.title           = element_text(size = 0), 
      legend.position        = "inside", 
      legend.position.inside = c(1, 1), 
      legend.justification   = c(1, 1), 
      legend.background      = element_rect(fill = alpha("white", alpha = 0.8))
    ) + 
    guides(
      linetype = guide_legend(
        override.aes = list(color = c("red", NA), linewidth = 0.5)
      ), 
      fill  = "none", 
      color = "none"
    ) + 
    coord_cartesian(
      xlim = c(x_1_min, x_1_max), # (目盛の共通化用)
      ylim = c(x_2_min, x_2_max), # (目盛の共通化用)
      ratio = 1
    ) + 
    labs(
      title = "multivariate Gaussian distribution", 
      subtitle = predict_param_lbl, 
      fill  = "density", 
      color = "density", 
      x = expression(x[1]), 
      y = expression(x[2])
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

  # ラベルを調整
  obs_param_lbl <- sprintf(
    fmt = "N == '%s'", 
    formatC(n, digits = 0, format = "d")
  ) |> 
    parse(text = _)
  tmp_model_graph <- model_graph + 
    labs(subtitle = obs_param_lbl)
  tmp_posterior_graph <- posterior_graph + 
    labs(subtitle = NULL)
  
  # グラフを並べて描画
  comb_graph <- cowplot::plot_grid(
    NULL,        NULL,            tmp_model_graph, 
    NULL,        NULL,            tmp_posterior_graph, 
    model_graph, posterior_graph, predict_graph, 
    nrow = 3, ncol = 3, 
    align = "hv" # (目盛の共通化用)
  )
  final_graph <- cowplot::plot_grid(
    title_graph, 
    comb_graph, 
    nrow = 2, ncol = 1, 
    rel_heights = c(0.05, 1)
  ) + 
    theme(
      plot.background = element_rect(fill = "white", color = NA) # (透過背景の対策用)
    )
  
  ##### グラフの出力 -----
  
  # 画像ファイルを書出
  file_path <- paste0(dir_path, "/", stringr::str_pad(n, width = nchar(N), pad = "0"), ".png")
  ggplot2::ggsave(
    filename = file_path, plot = final_graph, 
    width = 18, height = 18, units = "in", dpi = 100
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
    path = "figure/multivariate_gaussian_model/parameter_updates_mean/observation.mp4", 
    framerate = 10 
  ) -> tmp_path # mp4ファイルを書出


