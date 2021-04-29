
# 4.4.4 ガウス混合モデルにおける推論：崩壊型ギブスサンプリング -------------------------------------------

# 4.4.4項で利用するパッケージ
library(tidyverse)
library(mvnfast)


### 観測モデルの設定 -----

# 次元数を設定:(固定)
D <- 2

# クラスタ数を指定
K <- 3

# K個の真の平均を指定
mu_truth_kd <- matrix(
  c(5, 35, 
    -20, -10, 
    30, -20), nrow = K, ncol = D, byrow = TRUE
)

# K個の真の分散共分散行列を指定
sigma2_truth_ddk <- array(
  c(250, 65, 65, 270, 
    125, -45, -45, 175, 
    210, -15, -15, 250), dim = c(D, D, K)
)

# 真の混合比率を指定
pi_truth_k <- c(0.45, 0.25, 0.3)


# 作図用のx軸の値を作成
x_1_vec <- seq(
  min(mu_truth_kd[, 1] - 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  max(mu_truth_kd[, 1] + 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  length.out = 300)

# 作図用のy軸の値を作成
x_2_vec <- seq(
  min(mu_truth_kd[, 2] - 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  max(mu_truth_kd[, 2] + 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  length.out = 300
)

# 作図用のxの点を作成
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
  model_density <- model_density + tmp_density# * pi_truth_k[k]
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
       subtitle = paste0("K=", K), 
       x = expression(x[1]), y = expression(x[2]))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 250

# クラスタを生成
s_truth_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()

# クラスタ番号を取得
s_truth_n <- which(t(s_truth_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]

# (観測)データを生成
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


# 事前分布の平均による分布を計算
E_prior_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = mvnfast::dmvn(
    X = x_point_mat, mu = m_d, sigma = solve(nu * w_dd)
  )
)

# 事前分布の平均による分布を作図
ggplot(E_prior_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 事前分布の平均による分布
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0("E[mu]=(", paste0(m_d, collapse = ", "), 
                         "), E[lambda]=(", paste0(nu * w_dd, collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


### 初期値の設定 -----

# クラスタの初期値を生成
s_nk <- rmultinom(n = N, size = 1, prob = alpha_k / sum(alpha_k)) %>% 
  t()

# muの事後分布の初期値を計算:式(4.99)
beta_hat_k <- colSums(s_nk) + beta
m_hat_kd <- t(t(x_nd) %*% s_nk + beta * m_d) / beta_hat_k

# lambdaの事後分布の初期値を計算:式(4.103)
w_hat_ddk <- array(0, dim = c(D, D, K))
term_m_dd <- beta * matrix(m_d) %*% t(m_d)
for(k in 1:K) {
  term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
  term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
  w_hat_ddk[, , k] <- solve(term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd))
}
nu_hat_k <- colSums(s_nk) + nu

# piの事後分布の初期値を計算:式(4.45)
alpha_hat_k <- colSums(s_nk) + alpha_k


# 初期値による分布を計算
init_density <- 0
for(k in 1:K) {
  # クラスタkの確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # K個の確率密度の加重平均を計算
  init_density <- init_density + tmp_density #* pi_k[k]
}

# 初期値による分布をデータフレームに格納
init_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = init_density
)

# 真の平均をデータフレームに格納
mu_df <- tibble(
  x_1 = mu_truth_kd[, 1], 
  x_2 = mu_truth_kd[, 2]
)

# 初期値による分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = init_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 初期値による分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0("iter:", 0, ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


# 観測データとクラスタの初期値をデータフレームに格納
init_s_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"] %>% 
    as.factor()
)

# クラスタの初期値を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = init_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), alpha = 0.6) + # 初期値による分布
  geom_point(data = init_s_df, aes(x = x_n1, y = x_n2, color = cluster)) + # クラスタの初期値
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0("iter:", 0, ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


### 推論処理 -----

# 試行回数を指定
MaxIter <- 100

# 途中計算に用いる変数を作成
density_st_k <- rep(0, K)

# 推移の確認用の受け皿を作成
trace_s_in <- matrix(0, nrow = MaxIter + 1, ncol = N)
trace_E_mu_ikd <- array(0, dim = c(MaxIter + 1, K, D))
trace_E_lambda_iddk <- array(0, dim = c(MaxIter + 1, D, D, K))

# 初期値を記録
trace_s_in[1, ] <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]
trace_E_mu_ikd[1, , ] <- m_hat_kd
trace_E_lambda_iddk[1, , , ] <- rep(nu_hat_k, each = D * D) * w_hat_ddk

# 崩壊型ギブスサンプリング
for(i in 1:MaxIter) {
  for(n in 1:N) {
    
    # n番目のクラスタを初期化
    s_nk[n, ] <- rep(0, K)
    
    # muの事後分布のパラメータを計算:式(4.128)
    beta_hat_k <- colSums(s_nk) + beta
    m_hat_kd <- t(t(x_nd) %*% s_nk + beta * m_d) / beta_hat_k
    
    # lambdaの事後分布のパラメータを計算:式(4.128)
    w_hat_ddk <- array(0, dim = c(D, D, K))
    term_m_dd <- beta * matrix(m_d) %*% t(m_d)
    for(k in 1:K) {
      term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
      term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
      w_hat_ddk[, , k] <- solve(term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd))
    }
    nu_hat_k <- colSums(s_nk) + nu
    
    # piの事後分布のパラメータを計算:式(4.73)
    alpha_hat_k <- colSums(s_nk) + alpha_k
    
    # スチューデントのt分布のパラメータを計算
    nu_st_hat_k <- 1 - D + nu_hat_k
    m_st_hat_kd <- m_hat_kd
    term_lmd_k <- nu_st_hat_k * beta_hat_k / (1 + beta_hat_k)
    lambda_st_hat_ddk <- w_hat_ddk * array(rep(term_lmd_k, each = D * D), dim = c(D, D, K))
    
    # スチューデントのt分布の確率密度を計算:式(4.129)
    for(k in 1:K) {
      density_st_k[k] = mvnfast::dmvt(
        X = x_nd[n, ], mu = m_st_hat_kd[k, ], sigma = solve(lambda_st_hat_ddk[, , k]), df = nu_st_hat_k[k]
      ) + 1e-7
    }
    
    # カテゴリ分布のパラメータを計算:式(4.75)
    eta_k <- alpha_hat_k / sum(alpha_hat_k)
    
    # 潜在変数のサンプリング確率を計算:式(4.124)
    tmp_p_k <- density_st_k * eta_k
    p_s_k <- tmp_p_k / sum(tmp_p_k) # 正規化
    
    # n番目のデータのクラスタをサンプル
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = p_s_k) %>% 
      as.vector()
  }
  
  # muの事後分布の初期値を計算:式(4.99)
  beta_hat_k <- colSums(s_nk) + beta
  m_hat_kd <- t(t(x_nd) %*% s_nk + beta * m_d) / beta_hat_k
  
  # lambdaの事後分布の初期値を計算:式(4.103)
  w_hat_ddk <- array(0, dim = c(D, D, K))
  term_m_dd <- beta * matrix(m_d) %*% t(m_d)
  for(k in 1:K) {
    term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
    term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd))
  }
  nu_hat_k <- colSums(s_nk) + nu
  
  # piの事後分布の初期値を計算:式(4.45)
  alpha_hat_k <- colSums(s_nk) + alpha_k
  
  # i回目のクラスタとパラメータの期待値を記録
  trace_s_in[i + 1, ] <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"]
  trace_E_mu_ikd[i + 1, , ] <- m_hat_kd
  trace_E_lambda_iddk[i + 1, , , ] <- rep(nu_hat_k, each = D * D) * w_hat_ddk
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 推論結果を作図 -----

# 最終的なパラメータによる分布を計算
res_density <- 0
for(k in 1:K) {
  # クラスタkの確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # K個の確率密度の加重平均を計算
  res_density <- res_density + tmp_density #* pi_k[k]
}

# 最後的な分布をデータフレームに格納
res_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = res_density
)

# 最終的な分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = res_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 最終的な分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, ", K=", K), 
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
  geom_contour_filled(data = res_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), alpha = 0.6) + # 最終的な分布
  geom_point(data = s_df, aes(x = x_n1, y = x_n2, color = cluster)) + # サンプルしたクラスタ
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N), 
       x = expression(x[1]), y = expression(x[2]))


### パラメータの推移の確認 -----

# muの期待値の推移を作図
dplyr::as_tibble(trace_E_mu_ikd) %>% # データフレームに変換
  magrittr::set_names(
    paste0(rep(paste0("k=", 1:K), D), rep(paste0(", d=", 1:D), each = K))
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 0:MaxIter) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, 
    names_to = "dim", 
    values_to = "value"
  ) %>% # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
  geom_line() + 
  labs(title = "Collapsed Gibbs Sampling", 
       subtitle = expression(paste(E, "[", bold(mu), "]", sep = "")))

# lambdaのの期待値の推移を作図
dplyr::as_tibble(trace_E_lambda_iddk) %>% # データフレームに変換
  magrittr::set_names(
    paste0(
      rep(paste0("d=", 1:D), times = D * K), 
      rep(rep(paste0(", d=", 1:D), each = D), times = K), 
      rep(paste0(", k=", 1:K), each = D * D)
    )
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 0:MaxIter) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, 
    names_to = "dim", 
    values_to = "value"
  ) %>% # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
  geom_line() + 
  labs(title = "Collapsed Gibbs Sampling", 
       subtitle = expression(paste(E, "[", bold(Lambda), "]", sep = "")))


### gif画像で分布の推移を確認 -----

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # i回目の分布を計算
  res_density <- 0
  for(k in 1:K) {
    # クラスタkの確率密度を計算
    tmp_density <- mvnfast::dmvn(
      X = x_point_mat, 
      mu = trace_E_mu_ikd[i, k, ], 
      sigma = solve(trace_E_lambda_iddk[i, , , k])
    )
    
    # 確率密度加重平均を計算
    res_density <- res_density + tmp_density# * trace_E_pi_ik[i, k]
  }
  
  # i回目の分布をデータフレームに格納
  res_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = res_density, 
    iteration = as.factor(i - 1)
  )
  
  # 結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  
  # 観測データとi回目のサンプルしたクラスタをデータフレームに格納
  s_df <- tibble(
    x_n1 = x_nd[, 1], 
    x_n2 = x_nd[, 2], 
    cluster = as.factor(trace_s_in[i, ]), # 確率が最大のクラスタ番号
    iteration = as.factor(i - 1)
  )
  
  # 結果を結合
  trace_cluster_df <- rbind(trace_cluster_df, s_df)
  
  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}

# 分布の推移を作図
trace_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = trace_model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # i回目の分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  gganimate::transition_manual(iteration) + # フレーム
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}", ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


# クラスタの推移を作図
trace_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = trace_model_df, aes(x = x_1, y = x_2, z = density), alpha = 0.6) + # i回目の分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2, color = cluster)) + # i回目のクラスタ
  gganimate::transition_manual(iteration) + # フレーム
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}", ", N=", N), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter + 1, fps = 10)


