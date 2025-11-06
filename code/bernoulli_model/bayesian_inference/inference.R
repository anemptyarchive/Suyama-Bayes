
# ベルヌーイモデル -------------------------------------------------------------

# chapter 3.2.1
# ベイズ推論
# 推論アルゴリズムの実装


# ライブラリの読込 -------------------------------------------------------------

# 利用パッケージ
library(tidyverse)

# パッケージ名の省略用
library(ggplot2)


# ベイズ推論の実装 -------------------------------------------------------------

### ・生成分布(ベルヌーイ分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 0.25


# xがとり得る値を作成
x_vec <- 0:1

# 真の分布を計算:式(2.16)
model_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = c(1 - mu_truth, mu_truth) # 確率
)

# 真の分布を作図
ggplot() + 
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model"), 
           stat = "identity") + # 真の分布
  scale_fill_manual(breaks = "model", values = "purple", labels = "true model", name = "") + # バーの色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("mu==", mu_truth)), 
       x = "x", y = "probability")


### ・データの生成 -----

# (観測)データ数を指定
N <- 100


# ベルヌーイ分布に従うデータを生成
x_n <- rbinom(n = N, size = 1, prob = mu_truth)

# 観測データを集計
freq_df <- tibble::tibble(
  x = x_vec, # x軸の値
  freq = c(N - sum(x_n), sum(x_n)) # 度数
)

# 観測データのヒストグラムを作成
ggplot() + 
  geom_bar(data = freq_df, mapping = aes(x = x, y = freq/N, fill = "data"), 
           stat = "identity") + # 観測データ(相対度数)
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "model", color = "model"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, data = "pink"), na.value = NA, 
                    labels = c(model = "true model", data = "observation data"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", data = "pink"), 
                     labels = c(model = "true model", data = "observation data"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("list(mu==", mu_truth, ", N==", N, "(", paste0(freq_df[["freq"]], collapse = ", "), "))")), 
       x = "x", y = "relative frequency")


### ・事前分布(ベータ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


# グラフ用のmuの値を作成
mu_vec <- seq(0, 1, length.out = 501)

# 事前分布を計算:式(2.41)
prior_df <- tibble::tibble(
  mu = mu_vec, # x軸の値
  density = dbeta(x = mu_vec, shape1 = a, shape2 = b) # 確率密度
)

# 事前分布を作図
ggplot() + 
  geom_line(data = prior_df, mapping = aes(x = mu, y = density, color = "prior"), 
            size = 1) + # 事前分布
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", prior = "purple"), 
                     labels = c(param = "true parameter", prior = "prior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Beta Distribution", 
       subtitle = parse(text = paste0("list(a==", a, ", b==", b, ")")), 
       x = expression(mu), y = "density")


### ・事後分布(ベータ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.15)
a_hat <- sum(x_n) + a
b_hat <- N - sum(x_n) + b


# 事後分布を計算:式(2.41)
posterior_df <- tibble::tibble(
  mu = mu_vec, # x軸の値
  dens = dbeta(x = mu, shape1 = a_hat, shape2 = b_hat) # 確率密度
)

# 事後分布を作図
ggplot() + 
  geom_line(data = posterior_df, mapping = aes(x = mu, y = dens, color = "posterior"), 
            size = 1) + # 事後分布
  geom_vline(mapping = aes(xintercept = mu_truth, color = "param"), 
             size = 1, linetype = "dashed", show.legend = FALSE) + # 真のパラメータ
  scale_color_manual(values = c(param = "red", posterior = "purple"), 
                     labels = c(param = "true parameter", posterior = "posterior"), name = "") + # 線の色:(凡例表示用)
  guides(color = guide_legend(override.aes = list(size = c(0.8, 0.8), linetype = c(2, 1)))) + # 凡例の体裁:(凡例表示用)
  labs(title = "Beta Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(a)==", a_hat, ", hat(b)==", b_hat, ")")), 
       x = expression(mu), y = "density")


### ・予測分布(ベルヌーイ分布)の計算 -----

# 予測分布のパラメータを計算:式(3.19')
mu_star_hat <- a_hat / (a_hat + b_hat)
mu_star_hat <- (sum(x_n) + a) / (N + a + b)


# 予測分布を計算:式(2.16)
predict_df <- tibble::tibble(
  x = x_vec, # x軸の値
  prob = c(1 - mu_star_hat, mu_star_hat) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, mapping = aes(x = x, y = prob, fill = "predict"), 
           stat = "identity") + # 予測分布
  geom_bar(data = model_df, mapping = aes(x = x, y = prob, fill = "color", color = "color"), 
           stat = "identity", size = 1, linetype = "dashed") + # 真の分布
  scale_fill_manual(values = c(model = NA, predict ="purple"), na.value = NA, 
                    labels = c(model = "true model", predict = "predict"), name = "") + # バーの色:(凡例表示用)
  scale_color_manual(values = c(model = "red", predict ="purple"), 
                     labels = c(model = "true model", predict = "predict"), name = "") + # 線の色:(凡例表示用)
  scale_x_continuous(breaks = x_vec, minor_breaks = FALSE) + # x軸目盛
  ylim(c(0, 1)) + # y軸の表示範囲
  labs(title = "Bernoulli Distribution", 
       subtitle = parse(text = paste0("list(N==", N, ", hat(mu)[s]==", round(mu_star_hat, 2), ")")), 
       x = "x", y = "probability")


