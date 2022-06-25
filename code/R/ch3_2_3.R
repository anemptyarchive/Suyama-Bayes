
# 3.2.3 ポアソン分布の学習と予測 ---------------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)

# チェック用
library(ggplot2)
library(magrittr)


# ベイズ推論の実装 -------------------------------------------------------------------

### ・尤度(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth <- 4


# グラフ用のxの値を作成
x_vec <- seq(0, lambda_truth*4)

# 尤度を計算:式(2.37)
model_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = dpois(x = x, lambda = lambda_truth) # 確率
)

# 尤度を作図
ggplot() + 
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model"), 
           stat = "identity") + # 尤度
  scale_fill_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # バーの色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  labs(title = "Poisson Distribution", 
       subtitle = parse(text = paste0("lambda==", lambda_truth)), 
       x = "x", y = "probability")


### ・データの生成 -----

# (観測)データ数を指定
N <- 100


# ポアソン分布に従うデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)

# 観測データを集計
freq_df <- tidyr::tibble(x = x_n) |> # 観測データを格納
  dplyr::count(x, name = "freq") # 度数を集計

# ラベル用に度数を整形
freq_vec <- freq_df |> 
  dplyr::right_join(tidyr::tibble(x = x_vec), by = "x") |> # x軸の全ての値に結合
  dplyr::mutate(freq = tidyr::replace_na(freq, 0)) |> # 観測にない場合の欠損値を0に置換
  dplyr::pull(freq) # ベクトルとして抽出

# 観測データのヒストグラムを作成
ggplot() + 
  geom_bar(data = freq_df, mapping = aes(x = x, y = freq/N, fill = "data", color = "data"), 
           stat = "identity") + # 観測データ(相対度数)
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(1, 1), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  scale_x_continuous(breaks = 0:max(c(x_vec, x_n)), minor_breaks = FALSE) + # x軸目盛
  labs(title = "Poisson Distribution", 
       subtitle = parse(text = paste0("list(lambda==", lambda_truth, ", N==", N, "(", paste0(freq_vec, collapse = ", "), "))")), 
       x = "x", y = "relative frequency")


### ・事前分布(ガンマ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のlambdaの値を作成
lambda_vec <- seq(0, lambda_truth*2, length.out = 501)

# 事前分布を計算:式(2.56)
prior_df <- tibble::tibble(
  lambda = lambda_vec, # x軸の値
  dens = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# 事前分布を作図
ggplot() + 
  geom_line(data = prior_df, mapping = aes(x = lambda, y = dens, color = "prior"), 
            size = 1) + # 事前分布
  geom_vline(aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", prior = "purple"), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gamma Distribution", 
       subtitle = parse(text = paste0("list(a==", a, ", b==", b, ")"),), 
       x = expression(lambda), y = "density")


### 事後分布(ガンマ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.38)
a_hat <- sum(x_n) + a
b_hat <- N + b


# 事後分布を計算:式(2.56)
posterior_df <- tibble::tibble(
  lambda = lambda_vec, # x軸の値
  dens = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# 事後分布を作図
ggplot() + 
  geom_line(data = posterior_df, mapping = aes(x = lambda, y = dens, color = "posterior"), 
            size = 1) + # 事後分布
  geom_vline(mapping = aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", posterior = "purple"), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gamma Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(a)==", a_hat, ", hat(b)==", b_hat, ")")), 
       x = expression(lambda), y = "density")


### 予測分布(負の二項分布)の計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat <- a_hat
p_hat <- 1 / (b_hat + 1)
#r_hat <- sum(x_n) + a
#p_hat <- 1 / (N + b + 1)


# 予測分布を計算:式(3.43)
predict_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = dnbinom(x = x, size = r_hat, prob = 1-p_hat) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, mapping = aes(x = x, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, predict ="purple"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict ="purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  labs(title = "Negative Binomial Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(r)==", r_hat, ", hat(p)==", round(p_hat, 3), ")")), 
       x = "x", y = "probability")


# アニメーションによる学習推移の確認 ---------------------------------------------------------------

### ・モデルの設定 -----

# 真のパラメータを指定
lambda_truth <- 4

# 事前分布のパラメータを指定
a <- 1
b <- 1

# データ数(試行回数)を指定
N <- 300


# グラフ用のlambdaの値を作成
lambda_vec <- seq(0, lambda_truth*3, length.out = 501)

# グラフ用のxの値を作成
x_vec <- seq(0, lambda_truth*3)


### ・推論処理：for関数による処理 -----

# 事前分布(ガンマ分布)を計算
anime_posterior_df <- tibble::tibble(
  lambda = lambda_vec, # x軸の値
  dens = dgamma(x = lambda, shape = a, rate = b), # 確率密度
  param = paste0("N=", 0, "=(", paste0(rep(0, times = length(x_vec)), collapse = ", "), "), a=", a, ", b=", b) |> 
    as.factor() # フレーム切替用ラベル
)

# 初期値による予測分布のパラメータを計算:式(3.44)
r <- a
p <- 1 / (b + 1)

# 初期値による予測分布(負の二項分布)を計算
anime_predict_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = dnbinom(x = x, size = r, prob = 1 - p), # 確率
  param = paste0("N=", 0, "=(", paste0(rep(0, times = length(x_vec)), collapse = ", "), "), r=", r, ", p=", round(p, 5)) |> 
    as.factor() # フレーム切替用ラベル
)


# 観測データの受け皿を作成
x_n <- rep(NA, times = N)

# ベイズ推論
for(n in 1:N){
  
  # ポアソン分布に従うデータを生成
  x_n[n] <- rpois(n = 1 ,lambda = lambda_truth)
  
  # 観測データを集計
  freq_vec <- tidyr::tibble(x = x_n[1:n]) |> # 観測データを格納
    dplyr::count(x, name = "freq") |> # 度数を集計
    dplyr::right_join(tidyr::tibble(x = x_vec), by = "x") |> # x軸の全ての値に結合
    dplyr::mutate(freq = tidyr::replace_na(freq, 0)) |> # 観測にない場合の欠損値を0に置換
    dplyr::arrange(x) |> # 昇順に並べ替え
    dplyr::pull(freq) # ベクトルとして抽出
  
  # 事後分布のパラメータを更新:式(3.38)
  a <- sum(x_n[n] * 1) + a
  b <- 1 + b
  
  # 事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_df <- tibble::tibble(
    lambda = lambda_vec, # x軸の値
    dens = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    param = paste0("N=", n, "=(", paste0(freq_vec, collapse = ", "), "), a=", a, ", b=", b) |> 
      as.factor() # フレーム切替用ラベル
  )
  
  # 予測分布のパラメータを更新:式(3.44)
  r <- a
  p <- 1 / (b + 1)
  
  # 予測分布(負の二項分布)を計算:式(3.43)
  tmp_predict_df <- tibble::tibble(
    x = x_vec, # x軸の値
    prob = dnbinom(x = x, size = r, prob = 1-p), # 確率
    param = paste0("N=", n, "=(", paste0(freq_vec, collapse = ", "), "), r=", r, ", p=", round(p, 3)) |> 
      as.factor() # フレーム切替用のラベル
  )
  
  # n回目の結果を結合
  anime_posterior_df <- rbind(anime_posterior_df, tmp_posterior_df)
  anime_predict_df   <- rbind(anime_predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


### ・推論処理：tidyverseパッケージによる処理 -----

# ポアソン分布に従うデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)

# 試行ごとに度数を集計:(ラベル用)
freq_vec <- tibble::tibble(
  x = c(NA, x_n), # 観測データ
  n = 0:N, # 試行回数
  freq = 1 # 集計用の値
) |> # 観測データを格納
  dplyr::right_join(tidyr::expand_grid(x = x_vec, n = 0:N), by = c("x", "n")) |> # グラフ用の値に結合
  dplyr::mutate(freq = tidyr::replace_na(freq, replace = 0)) |> # 観測にない場合の欠損値を0に置換
  dplyr::arrange(n, x) |> # 集計用に昇順に並べ替え
  dplyr::group_by(x) |> # 集計用にグループ化
  dplyr::mutate(freq = cumsum(freq)) |> # 各試行までの度数を計算
  dplyr::ungroup() |> # グループ化を解除
  tidyr::pivot_wider(
    id_cols = n, 
    names_from = x, 
    names_prefix = "x", 
    values_from = freq
  ) |> # 度数列を展開
  tidyr::unite(col = "freq", dplyr::starts_with("x"), sep = ", ") |> # 度数情報をまとめて文字列化
  dplyr::pull(freq) # ベクトルとして抽出

# 試行ごとに事後分布(ガンマ分布)を計算
anime_posterior_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  lambda = lambda_vec # x軸の値
) |> 
  dplyr::mutate(
    a = c(a, cumsum(x_n) + a)[n+1], 
    b = n + b
  ) |> # 事後分布のパラメータを計算:式(3.38)
  dplyr::mutate(
    dens = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    param = paste0("N=", n, "=(", freq_vec[n+1], "), a=", a, ", b=", b) %>% 
      factor(., levels = unique(.)) # フレーム切替用ラベル
  ) # 事後分布を計算:式(2.56)

# 試行ごとに予測分布(負の二項分布)を計算
anime_predict_df <- tidyr::expand_grid(
  n = 0:N, # 試行回数
  x = x_vec # x軸の値
) |> 
  dplyr::mutate(
    r = c(a, cumsum(x_n) + a)[n+1], 
    p = 1 / (n + b + 1)
  ) |> # 予測分布のパラメータを計算算:式(3.44)
  dplyr::mutate(
    prob = dnbinom(x = x, size = r, prob = 1-p), # 確率
    param = paste0("N=", n, "=(", freq_vec[n+1], "), r=", r, ", p=", round(p, 3)) %>% 
      factor(., levels = unique(.)) # フレーム切替用ラベル
  ) # 予測分布を計算:式(3.43)


### ・作図処理 -----

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_posterior_df[["param"]]) # フレーム切替用ラベル
)

# 事後分布のアニメーションを作図
posterior_graph <- ggplot() + 
  geom_vline(mapping = aes(xintercept = lambda_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  geom_line(data = anime_posterior_df, mapping = aes(x = lambda, y = dens, color = "posterior"), 
            size = 1) + # 事後分布
  geom_point(data = anime_data_df, mapping = aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_color_manual(breaks = c("param", "posterior", "data"), 
                     values = c("red", "purple", "pink"), 
                     labels = c("true parameter", "posterior", "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.5, 0.5, 5), linetype = c(2, 1, 0), shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda), y = "density")

# gif画像を作成
gganimate::animate(posterior_graph, nframes = N+1, fps = 10, width = 800, height = 600)


# 尤度をフレーム分に複製
anime_model_df <- tibble::tibble(
  x = rep(x_vec, times = N+1), # x軸の値
  prob = rep(dpois(x = x_vec, lambda = lambda_truth), times = N+1), # 確率
  param = anime_predict_df[["param"]] # フレーム切替用ラベル
)

# 観測データを格納
anime_data_df <- tibble::tibble(
  x = c(NA, x_n), 
  param = unique(anime_predict_df[["param"]]) # フレーム切替用ラベル
)

# 予測分布のアニメーションを作図
predict_graph <- ggplot() + 
  geom_bar(data = anime_predict_df, mapping = aes(x = x, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = anime_model_df, aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  geom_point(data = anime_data_df, aes(x = x, y = 0, color = "data"), 
             size = 6) + # 観測データ
  gganimate::transition_manual(param) + # フレーム
  scale_fill_manual(values = c(model = NA, predict = "purple", data = "pink"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict = "purple", data = "pink"), 
                     labels = c(model = "true model", predict = "predict", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  guides(fill = guide_legend(override.aes = list(fill = c(NA, "purple", NA))), 
         color = guide_legend(override.aes = list(size = c(0.5, 0.5, 5), linetype = c(2, 0, 0), shape = c(NA, NA, 19)))) + # 凡例の体裁:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  coord_cartesian(ylim = c(0, 0.5)) + # 軸の表示範囲
  labs(title = "Negative Binomial Distribution", 
       subtitle = "{current_frame}", 
       x = "x", y = "probability")

# gif画像を作成
gganimate::animate(predict_graph, nframes = N+1, fps = 10, width = 800, height = 600)


