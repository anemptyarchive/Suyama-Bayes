
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


# 作図用のx軸のxの値を作成
x_1_vec <- seq(
  min(mu_truth_kd[, 1] - 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  max(mu_truth_kd[, 1] + 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  length.out = 300)

# 作図用のy軸のxの値を作成
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
  # クラスタkの分布の確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k]
  )
  
  # K個の分布の加重平均を計算
  model_density <- model_density + pi_truth_k[k] * tmp_density
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

# 潜在変数を生成
s_truth_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()

# クラスタ番号を取得
s_truth_n <- which(t(s_truth_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]

# (観測)データを生成
x_nd <- matrix(0, nrow = N, ncol = D)
for(n in 1:N) {
  # n番目のデータのクラスタ番号を取得
  k <- s_truth_n[n]
  
  # n番目のデータを生成
  x_nd[n, ] = mvnfast::rmvn(n = 1, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k])
}

# 観測データと真のクラスタ番号をデータフレームに格納
x_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = as.factor(s_truth_n)
)

# 観測データの散布図を作成
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               linetype = "dashed") + # 真の分布
  #geom_contour_filled(data = model_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), 
  #                    alpha = 0.6, linetype = "dashed") + # 真の分布:(塗りつぶし)
  geom_point(data = x_df, aes(x = x_n1, y = x_n2, color = cluster)) + # 真のクラスタ
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0('N=', N, ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


### 事前分布(ガウス・ウィシャート分布)の設定 -----

# muの事前分布のパラメータを指定
beta <- 1
m_d <- rep(0, D)

# lambdaの事前分布のパラメータを指定
w_dd <- diag(D) * 0.0005
nu <- D
sqrt(solve(beta * nu * w_dd)) # (似非)相関行列の平均

# piの事前分布のパラメータを指定
alpha_k <- rep(2, K)


# 真の平均をデータフレームに格納
mu_df <- tibble(
  x_1 = mu_truth_kd[, 1], 
  x_2 = mu_truth_kd[, 2]
)

# muの事前分布の標準偏差を計算
sigma_mu_d <- solve(beta * nu * w_dd) %>% 
  diag() %>% 
  sqrt()

# 作図用のx軸のmuの値を作成
mu_1_vec <- seq(
  min(mu_truth_kd[, 1]) - sigma_mu_d[1], 
  max(mu_truth_kd[, 1]) + sigma_mu_d[1], 
  length.out = 250)

# 作図用のy軸のmuの値を作成
mu_2_vec <- seq(
  min(mu_truth_kd[, 2]) - sigma_mu_d[2], 
  max(mu_truth_kd[, 2]) + sigma_mu_d[2], 
  length.out = 250
)

# 作図用のmuの点を作成
mu_point_mat <- cbind(
  rep(mu_1_vec, times = length(mu_2_vec)), 
  rep(mu_2_vec, each = length(mu_1_vec))
)

# muの事前分布をデータフレームに格納
prior_mu_df <- tibble(
  mu_1 = mu_point_mat[, 1], 
  mu_2 = mu_point_mat[, 2], 
  density = mvnfast::dmvn(X = mu_point_mat, mu = m_d, sigma = solve(beta * nu * w_dd))
)

# muの事前分布を作図
ggplot() + 
  geom_contour(data = prior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = ..level..)) + # 事前分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("m=(", paste0(m_d, collapse = ", "), ")", 
                         ", sigma2_mu=(", paste0(solve(beta * nu * w_dd), collapse = ", "), ")"), 
       x = expression(mu[1]), y = expression(mu[2]))


### 初期値の設定 -----

# 潜在変数を初期化
s_nk <- rmultinom(n = N, size = 1, prob = alpha_k / sum(alpha_k)) %>% 
  t()

# 初期値によるmuの事後分布のパラメータを計算:式(4.99)
beta_hat_k <- colSums(s_nk) + beta
m_hat_kd <- t(t(x_nd) %*% s_nk + beta * m_d) / beta_hat_k

# 初期値によるlambdaの事後分布のパラメータを計算:式(4.103)
w_hat_ddk <- array(0, dim = c(D, D, K))
term_m_dd <- beta * matrix(m_d) %*% t(m_d)
for(k in 1:K) {
  term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
  term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
  w_hat_ddk[, , k] <- solve(term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd))
}
nu_hat_k <- colSums(s_nk) + nu

# 初期値によるpiの事後分布のパラメータを計算:式(4.45)
alpha_hat_k <- colSums(s_nk) + alpha_k


# 初期値によるmuの事後分布をデータフレームに格納
posterior_mu_df <- tibble()
for(k in 1:K) {
  # クラスタkのmuの事後分布を計算
  tmp_density <- mvnfast::dmvn(
    X = mu_point_mat, 
    mu = m_hat_kd[k, ], 
    sigma = solve(beta_hat_k[k] * nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # クラスタkの分布をデータフレームに格納
  tmp_df <- tibble(
    mu_1 = mu_point_mat[, 1], 
    mu_2 = mu_point_mat[, 2], 
    density = tmp_density, 
    cluster = as.factor(k)
  )
  
  # 結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_df)
}

# 初期値によるmuの事後分布を作図
ggplot() + 
  geom_contour(data = posterior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = cluster)) + # 事後分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("iter:", 0, ", N=", N), 
       x = expression(mu[1]), y = expression(mu[2]))


# 初期値による混合分布を計算
init_density <- 0
for(k in 1:K) {
  # クラスタkの分布の確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # K個の分布の加重平均を計算
  init_density <- init_density + alpha_hat_k[k] / sum(alpha_hat_k) * tmp_density
}

# 初期値による分布をデータフレームに格納
init_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = init_density
)

# 初期値による分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_contour(data = init_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 初期値による分布
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0("iter:", 0, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


# 観測データとクラスタの初期値をデータフレームに格納
s_df <- tibble(
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
  geom_contour_filled(data = init_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), 
                      alpha = 0.6) + # 初期値による分布
  geom_point(data = s_df, aes(x = x_n1, y = x_n2, color = cluster)) + # クラスタの初期値
  labs(title = "Gaussian Mixture Model", 
       subtitle = paste0("iter:", 0, ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


### 推論処理 -----

# 試行回数を指定
MaxIter <- 150

# 途中計算に用いる変数を作成
density_st_k <- rep(0, K)

# 推移の確認用の受け皿を初期化
trace_s_in     <- matrix(0, nrow = MaxIter + 1, ncol = N)
trace_beta_ik  <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_m_ikd    <- array(0, dim = c(MaxIter + 1, K, D))
trace_w_iddk   <- array(0, dim = c(MaxIter + 1, D, D, K))
trace_nu_ik    <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_alpha_ik <- matrix(0, nrow = MaxIter + 1, ncol = K)

# 初期値を記録
trace_s_in[1, ]       <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]
trace_beta_ik[1, ]    <- beta_hat_k
trace_m_ikd[1, , ]    <- m_hat_kd
trace_w_iddk[1, , , ] <- w_hat_ddk
trace_nu_ik[1, ]      <- nu_hat_k
trace_alpha_ik[1, ]   <- alpha_hat_k

# 崩壊型ギブスサンプリング
for(i in 1:MaxIter) {
  for(n in 1:N) {
    
    # n番目のデータの潜在変数を初期化
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
    mu_st_hat_kd <- m_hat_kd
    term_lmd_k <- nu_st_hat_k * beta_hat_k / (1 + beta_hat_k)
    lambda_st_hat_ddk <- w_hat_ddk * array(rep(term_lmd_k, each = D * D), dim = c(D, D, K))
    
    # スチューデントのt分布の確率密度を計算:式(4.129)
    for(k in 1:K) {
      density_st_k[k] = mvnfast::dmvt(
        X = x_nd[n, ], mu = mu_st_hat_kd[k, ], sigma = solve(lambda_st_hat_ddk[, , k]), df = nu_st_hat_k[k]
      ) + 1e-7
    }
    
    # カテゴリ分布のパラメータを計算:式(4.75)
    eta_k <- alpha_hat_k / sum(alpha_hat_k)
    
    # 潜在変数のサンプリング確率を計算:式(4.124)
    tmp_prob_k <- density_st_k * eta_k
    prob_s_k <- tmp_prob_k / sum(tmp_prob_k) # 正規化
    
    # n番目のデータの潜在変数をサンプル
    s_nk[n, ] <- rmultinom(n = 1, size = 1, prob = prob_s_k) %>% 
      as.vector()
  }
  
  # muの事後分布のパラメータを計算:式(4.99)
  beta_hat_k <- colSums(s_nk) + beta
  m_hat_kd <- t(t(x_nd) %*% s_nk + beta * m_d) / beta_hat_k
  
  # lambdaの事後分布のパラメータを計算:式(4.103)
  w_hat_ddk <- array(0, dim = c(D, D, K))
  term_m_dd <- beta * matrix(m_d) %*% t(m_d)
  for(k in 1:K) {
    term_x_dd <- t(s_nk[, k] * x_nd) %*% x_nd
    term_m_hat_dd <- beta_hat_k[k] * matrix(m_hat_kd[k, ]) %*% t(m_hat_kd[k, ])
    w_hat_ddk[, , k] <- solve(term_x_dd + term_m_dd - term_m_hat_dd + solve(w_dd))
  }
  nu_hat_k <- colSums(s_nk) + nu
  
  # piの事後分布のパラメータを計算:式(4.45)
  alpha_hat_k <- colSums(s_nk) + alpha_k
  
  # i回目のパラメータを記録
  trace_s_in[i + 1, ]       <- which(t(s_nk) == 1, arr.ind = TRUE) %>% 
    .[, "row"]
  trace_beta_ik[i + 1, ]    <- beta_hat_k
  trace_m_ikd[i + 1, , ]    <- m_hat_kd
  trace_w_iddk[i + 1, , , ] <- w_hat_ddk
  trace_nu_ik[i + 1, ]      <- nu_hat_k
  trace_alpha_ik[i + 1, ]   <- alpha_hat_k
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 推論結果を作図 -----

# muの事後分布をデータフレームに格納
posterior_mu_df <- tibble()
for(k in 1:K) {
  # クラスタkのmuの事後分布を計算
  tmp_density <- mvnfast::dmvn(
    X = mu_point_mat, 
    mu = m_hat_kd[k, ], 
    sigma = solve(beta_hat_k[k] * nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # クラスタkの分布をデータフレームに格納
  tmp_df <- tibble(
    mu_1 = mu_point_mat[, 1], 
    mu_2 = mu_point_mat[, 2], 
    density = tmp_density, 
    cluster = as.factor(k)
  )
  
  # 結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_df)
}

# muの事後分布を作図
ggplot() + 
  geom_contour(data = posterior_mu_df, aes(x = mu_1, y = mu_2, z = density, color = cluster)) + # 事後分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian Distribution:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N), 
       x = expression(mu[1]), y = expression(mu[2]))


# 最後に更新したパラメータの期待値による混合分布を計算
res_density <- 0
for(k in 1:K) {
  # クラスタkの分布の確率密度を計算
  tmp_density <- mvnfast::dmvn(
    X = x_point_mat, mu = m_hat_kd[k, ], sigma = solve(nu_hat_k[k] * w_hat_ddk[, , k])
  )
  
  # K個の分布の加重平均を計算
  res_density <- res_density + alpha_hat_k[k] / sum(alpha_hat_k) * tmp_density
}

# 最終的な分布をデータフレームに格納
res_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = res_density
)

# 最終的な分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = res_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 期待値による分布
  geom_point(data = x_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


# 観測データと最後にサンプルしたクラスタ番号をデータフレームに格納
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
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = res_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), 
                      alpha = 0.6) + # 期待値による分布
  geom_point(data = s_df, aes(x = x_n1, y = x_n2, color = cluster)) + # サンプルしたクラスタ
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:", MaxIter, ", N=", N, 
                         ", E[pi]=(", paste0(round(alpha_hat_k / sum(alpha_hat_k), 3), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


### 超パラメータの推移の確認 -----

# mの推移を作図
dplyr::as_tibble(trace_m_ikd) %>% # データフレームに変換
  magrittr::set_names(
    paste0(rep(paste0("k=", 1:K), D), rep(paste0(", d=", 1:D), each = K))
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "dim", # 現列名を格納する列名
    values_to = "value" # 現要素を格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
    geom_line() + 
    labs(title = "Collapsed Gibbs Sampling", 
         subtitle = expression(hat(m)))

# betaの推移を作図
dplyr::as_tibble(trace_beta_ik) %>% # データフレームに変換
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = cluster)) + 
    geom_line() + # 推定値
    labs(title = "Collapsed Gibbs Sampling", 
         subtitle = expression(hat(beta)))

# nuの推移を作図
dplyr::as_tibble(trace_nu_ik) %>% # データフレームに変換
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = cluster)) + 
    geom_line() + # 推定値
    labs(title = "Collapsed Gibbs Sampling", 
         subtitle = expression(hat(nu)))

# wの推移を作図
dplyr::as_tibble(trace_w_iddk) %>% # データフレームに変換
  magrittr::set_names(
    paste0(
      rep(paste0("d=", 1:D), times = D * K), 
      rep(rep(paste0(", d=", 1:D), each = D), times = K), 
      rep(paste0(", k=", 1:K), each = D * D)
    )
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "dim", # 現列名を格納する列名
    values_to = "value" # 現要素を格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
    geom_line(alpha = 0.5) + 
    labs(title = "Collapsed Gibbs Sampling", 
         subtitle = expression(hat(w)))

# alphaの推移を作図
dplyr::as_tibble(trace_alpha_ik) %>% # データフレームに変換
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = cluster)) + 
    geom_line() + # 推定値
    labs(title = "Collapsed Gibbs Sampling", 
         subtitle = expression(hat(alpha)))


# おまけ：アニメーションによる推移を確認 -----------------------------------------------------

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
trace_posterior_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  for(k in 1:K) {
    # クラスタkのmuの事後分布を計算
    tmp_density <- mvnfast::dmvn(
      X = mu_point_mat, 
      mu = trace_m_ikd[i, k, ], 
      sigma = solve(trace_beta_ik[i, k] * trace_nu_ik[i, k] * trace_w_iddk[i, , , k])
    )
    
    # クラスタkの分布をデータフレームに格納
    tmp_df <- tibble(
      mu_1 = mu_point_mat[, 1], 
      mu_2 = mu_point_mat[, 2], 
      density = tmp_density, 
      cluster = as.factor(k), 
      iteration = as.factor(i - 1)
    )
    
    # 結果を結合
    trace_posterior_df <- rbind(trace_posterior_df, tmp_df)
  }
  
  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}

# muの事後分布を作図
trace_posterior_graph <- ggplot() + 
  geom_contour(data = trace_posterior_df, aes(x = mu_1, y = mu_2, z = density, color = cluster)) + # 事後分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の値
  gganimate::transition_manual(iteration) + # フレーム
  labs(title = "Gaussian Distribution:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}", ", N=", N), 
       x = expression(mu[1]), y = expression(mu[2]))

# gif画像を作成
gganimate::animate(trace_posterior_graph, nframes = MaxIter + 1, fps = 10)


# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # i回目の混合分布を計算
  res_density <- 0
  for(k in 1:K) {
    # クラスタkの分布の確率密度を計算
    tmp_density <- mvnfast::dmvn(
      X = x_point_mat, 
      mu = trace_m_ikd[i, k, ], 
      sigma = solve(trace_nu_ik[i, k] * trace_w_iddk[i, , , k])
    )
    
    # K個の分布の加重平均を計算
    res_density <- res_density + trace_alpha_ik[i, k] / sum(trace_alpha_ik[i, ]) * tmp_density
  }
  
  # i回目の分布をデータフレームに格納
  res_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = res_density, 
    label = paste0(
      "iter:", i - 1, ", N=", N, 
      ", pi=(", paste0(round(trace_alpha_ik[i, ] / sum(trace_alpha_ik[i, ]), 3), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  
  # 観測データとi回目のクラスタ番号をデータフレームに格納
  s_df <- tibble(
    x_n1 = x_nd[, 1], 
    x_n2 = x_nd[, 2], 
    cluster = as.factor(trace_s_in[i, ]), 
    label = paste0(
      "iter:", i - 1, ", N=", N, 
      ", pi=(", paste0(round(trace_alpha_ik[i, ] / sum(trace_alpha_ik[i, ]), 3), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 結果を結合
  trace_cluster_df <- rbind(trace_cluster_df, s_df)
  
  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}


# 分布の推移を作図
trace_model_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = trace_model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 期待値による分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = paste0("iter:{current_frame}"), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_model_graph, nframes = MaxIter + 1, fps = 10)


# クラスタの推移を作図
trace_cluster_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = trace_model_df, aes(x = x_1, y = x_2, z = density), 
                      alpha = 0.6) + # 期待値による分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2, color = cluster)) + # サンプルしたクラスタ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian Mixture Model:Collapsed Gibbs Sampling", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_cluster_graph, nframes = MaxIter + 1, fps = 10)


