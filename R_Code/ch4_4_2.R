
# 4.4.2 ガウス混合モデルにおける推論：ギブスサンプリング -------------------------------------------

# 4.4.2項で利用するパッケージ
library(tidyverse)
library(mvnfast)
library(MCMCpack)


### 観測モデルの設定 -----

# 次元数を設定:(固定)
D <- 2

# クラスタ数を指定
K <- 3

# 真の平均を指定
mu_truth_kd <- matrix(
  c(5, 35, 
    -20, -10, 
    30, -20), nrow = K, ncol = D, byrow = TRUE
)

# 真の分散共分散行列を指定
sigma2_truth_ddk <- array(
  c(250, 65, 65, 270, 
    125, -45, -45, 175, 
    210, -15, -15, 250), dim = c(D, D, K)
)

# 真の混合比率を指定
pi_truth_k <- c(0.45, 0.25, 0.3)


# 作図用の点を生成
x_1_vec <- seq(
  min(mu_truth_kd[, 1] - 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  max(mu_truth_kd[, 1] + 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  length.out = 300)
x_2_vec <- seq(
  min(mu_truth_kd[, 2] - 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  max(mu_truth_kd[, 2] + 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  length.out = 300
)
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)

# 観測モデルを計算
model_density <- 0
for(k in 1:K) {
  # クラスタkの確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k]
  )
  
  # K個の確率密度の加重平均を計算
  model_density <- model_density + tmp_density * pi_truth_k[k]
}

# 観測モデルをデータフレームに格納
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = model_density
)

# 観測モデルを作図
ggplot(model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 真の分布
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('K=', K), 
       x = expression(x[1]), y = expression(x[2]))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 250

# クラスタを生成
s_truth_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()
s_truth_n <- which(t(s_truth_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]

# 観測データを生成
x_nd <- matrix(0, nrow = N, ncol = D)
for(n in 1:N) {
  # n番目のデータのクラスタを取得
  k <- s_truth_n[n]
  
  # n番目のデータを生成
  x_nd[n, ] = mvnfast::rmvn(n = 1, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k])
}

# 観測データと真のクラスタをデータフレームに格納
x_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = as.factor(s_truth_n)
)

# 観測データの散布図を作成
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               linetype = "dashed") + # 真の分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2, color = cluster)) + # 真のクラスタ
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('N=', N, ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


### 事前分布(ガウス・ウィシャート分布)の設定 -----

# muの事前分布のパラメータを指定
beta <- 1
m_d <- rep(0, D)

# lambdaの事前分布のパラメータを指定
nu <- D
w_dd <- diag(D) * 0.0005
sqrt(solve(beta * nu * w_dd)) # (似非)相関行列の平均

# piの事前分布のパラメータを指定
alpha_k <- rep(1, K)


# 観測モデルのパラメータをサンプル
mu_kd <- matrix(0, nrow = K, ncol = D)
lambda_ddk <- array(0, dim = c(D, D, K))
for(k in 1:K) {
  # クラスタkの精度行列をサンプル
  lambda_ddk[, , k] <- rWishart(n = 1, df = nu, Sigma = w_dd)
  
  # クラスタkの平均をサンプル
  mu_kd[k, ] <- mvnfast::rmvn(n = 1, mu = m_d, sigma = solve(beta * lambda_ddk[, , k]))
}

# 混合比率をサンプル
pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_k) %>% 
  as.vector()


# 事前分布からサンプルした分布を計算
init_density <- 0
for(k in 1:K) {
  # クラスタkの確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_kd[k, ], sigma = solve(lambda_ddk[, , k])
  )
  
  # K個の確率密度の加重平均を計算
  init_density <- init_density + tmp_density * pi_k[k]
}

# 事前分布からサンプルしたパラメータによる分布データフレームに格納
init_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = init_density
)

# 事前分布からサンプルしたパラメータによる分布を作図
ggplot() + 
  geom_contour(data = init_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # サンプルした分布
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('K=', K), 
       x = expression(x[1]), y = expression(x[2]))


### 推論処理 -----

# 試行回数を指定
MaxIter <- 500

# 推移の確認用の受け皿を初期化
trace_s_in <- matrix(0, nrow = MaxIter, ncol = N)
trace_mu_ikd <- array(0, dim = c(MaxIter, K, D))
trace_lambda_iddk <- array(0, dim = c(MaxIter, D, D, K))

# ギブスサンプリング
for(i in 1:MaxIter) {
  
  # パラメータを初期化
  eta_nk <- matrix(0, nrow = N, ncol = K)
  s_nk <- matrix(0, nrow = N, ncol = K)
  beta_hat_k <- rep(0, K)
  m_hat_kd <- matrix(0, nrow = K, ncol = D)
  w_hat_ddk <- array(0, dim = c(D, D, K))
  nu_hat_k <- rep(0, K)
  alpha_hat_k <- rep(0, K)
  
  # 潜在変数のパラメータを計算:式(4.94)
  for(k in 1:K) {
    tmp_x_dn <- t(x_nd) - mu_kd[k, ]
    term_x_n <- (t(tmp_x_dn) %*% lambda_ddk[, , k] %*% tmp_x_dn) %>% 
      diag()
    term_ln <- 0.5 * log(det(lambda_ddk[, , k]) + 1e-7) + log(pi_k[k] + 1e-7)
    eta_nk[, k] <- exp(term_ln - 0.5 * term_x_n)
  }
  eta_nk <- eta_nk / rowSums(eta_nk) # 正規化
  
  # 潜在変数をサンプル:式(4.39)
  for(n in 1:N) {
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
  }
  
  # 観測モデルのパラメータをサンプル
  for(k in 1:K) {
    
    # muの事後分布のパラメータを計算:式(4.99)
    beta_hat_k[k] <- sum(s_nk[, k]) + beta
    m_hat_kd[k, ] <- (colSums(s_nk[, k] * x_nd) + beta * m_d) / beta_hat_k[k]
    
    # lambdaの事後分布のパラメータを計算:式(4.103)
    term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
    term_m_dd <- beta * matrix(m_d) %*% t(m_d)
    term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(
      term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd)
    )
    nu_hat_k[k] <- sum(s_nk[, k]) + nu
    
    # lambdaをサンプル:式(4.102)
    lambda_ddk[, , k] <- rWishart(n = 1, df = nu_hat_k[k], Sigma = w_hat_ddk[, , k])

    # muをサンプル:式(4.98)
    mu_kd[k, ] <- mvnfast::rmvn(
      n = 1, mu = m_hat_kd[k, ], sigma = solve(beta_hat_k[k] * lambda_ddk[, , k])
    ) %>% 
      as.vector()
  }
  
  # 混合比率の事後分布のパラメータを計算:式(4.45)
  alpha_hat_k <- colSums(s_nk) + alpha_k
  
  # piをサンプル:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_hat_k) %>% 
    as.vector()
  
  # パラメータを記録
  trace_s_in[i, ] <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"]
  trace_mu_ikd[i, , ] <- mu_kd
  trace_lambda_iddk[i, , , ] <- lambda_ddk
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 分布を作図 -----

# 最後にサンプルしたパラメータによる分布を計算
res_density <- 0
for(k in 1:K) {
  # クラスタkの確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_kd[k, ], sigma = solve(lambda_ddk[, , k])
  )
  
  # K個の確率密度の加重平均を計算
  res_density <- res_density + tmp_density * pi_k[k]
}

# 最後的な分布をデータフレームに格納
res_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = res_density
)

# 真の平均をデータフレームに格納
mu_df <- tibble(
  x_1 = mu_truth_kd[, 1], 
  x_2 = mu_truth_kd[, 2]
)

# 最終的な分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = res_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # サンプルした分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Gaussian Mixture Model:Gibbs Sampling", 
       subtitle = paste0("N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


# 観測データとサンプルしたクラスタをデータフレームに格納
s_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"] %>% 
    as.factor()
)

# 最終的なクラスタを作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = res_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), alpha = 0.6) + # サンプルした分布
  geom_point(data = s_df, aes(x = x_n1, y = x_n2, color = cluster)) + # サンプルしたクラスタ
  labs(title = "Gaussian Mixture Model:Gibbs Sampling", 
       subtitle = paste0("N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


### パラメータの推移を確認 -----

# muの推移を作図
as_tibble(trace_mu_ikd) %>% # データフレームに変換
  magrittr::set_names(
    paste0(rep(paste0("k=", 1:K), D), rep(paste0(", d=", 1:D), each = K))
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:MaxIter) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, 
    names_to = "dim", 
    values_to = "value"
  ) %>% # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
    geom_line() + 
    labs(title = "Gibbs Sampling", 
         subtitle = expression(mu))

# lambdaの推移を作図
as_tibble(trace_lambda_iddk) %>% # データフレームに変換
  magrittr::set_names(
    paste0(
      rep(paste0("d=", 1:D), times = D * K), 
      rep(rep(paste0(", d=", 1:D), each = D), times = K), 
      rep(paste0(", k=", 1:K), each = D * D)
    )
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:MaxIter) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, 
    names_to = "dim", 
    values_to = "value"
  ) %>% # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
    geom_line(alpha = 0.5) + 
    labs(title = "Gibbs Sampling", 
         subtitle = expression(Lambda))


### gif画像で分布の推移を確認 -----

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:MaxIter) {
  # i回目の分布を計算
  res_density <- 0
  for(k in 1:K) {
    # クラスタkの確率密度を計算
    tmp_density <- mvnfast::dmvn(
      X = x_point_mat, 
      mu = trace_mu_ikd[i, k, ], 
      sigma = solve(trace_lambda_iddk[i, , , k])
    )
    
    # 確率密度加重平均を計算
    res_density <- res_density + tmp_density * pi_k[k]
  }
  
  # i回目の分布をデータフレームに格納
  res_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = res_density, 
    iteration = as.factor(i)
  )
  
  # 結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  
  # 観測データとi回目のサンプルしたクラスタをデータフレームに格納
  s_df <- tibble(
    x_n1 = x_nd[, 1], 
    x_n2 = x_nd[, 2], 
    cluster = as.factor(trace_s_in[i, ]), 
    iteration = as.factor(i)
  )
  
  # 結果を結合
  trace_cluster_df <- rbind(trace_cluster_df, s_df)
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}

# 分布の推移を作図
trace_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = trace_model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # サンプルした分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  gganimate::transition_manual(iteration) + # フレーム
  labs(title = "Gaussian Mixture Model:Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}", ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter, fps = 10)


# クラスタの推移を作図
trace_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = trace_model_df, aes(x = x_1, y = x_2, z = density), alpha = 0.6) + # サンプルした分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2, color = cluster)) + # サンプル↓クラスタ
  gganimate::transition_manual(iteration) + # フレーム
  labs(title = "Gaussian Mixture Model:Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}", ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter, fps = 10)


