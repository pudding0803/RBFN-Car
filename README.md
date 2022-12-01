# RBFN Car
> 類神經網路的 project 2

---

## 整體流程
1. 使用者輸入 `4` 或 `6` 代表使用的訓練集（train`?`dAll.txt），這次懶得做 GUI 😅，訓練資料比例為 100% 🥴
2. 用 K-means 進行 RBFN 神經元的初始化（非監督式學習），預設分成 $15$ 群（或更少），結束條件為「與上次分群結果完全相同」，若輸入 `4` 會利用 plotly 函式庫做 Data Visualization
3. 每次輸入 $x$，前傳遞算出 $F(x)$ 後，倒傳遞調整參數（監督式學習），參數包含 bias( $\theta$ )、權重 ( $w$ )、中心 ( $m$ )、標準差 ( $\sigma$ )
    * $eta$ 的調整
        * 這次 $eta$ 使用隨著 $epoch$ 線性遞減的方式，希望前期收斂速度不要太慢，且後期能更容易收斂成功， $eta = \frac{epoch - e}{epoch} \times 0.1$
    * Forward Propagation
        * Activation function 使用 $\phi(x) = exp(-\frac{|x-m|^2}{2\sigma^2})$
        * $F(x) = \sum{(x_j \cdot w_j(n))} + \theta(n)$
    * Backward Propagation
        * $\theta(n+1) = \theta(n) + \eta \cdot (y - F(x))$
        * $w_j(n+1) = w_j(n) + \eta \cdot (y - F(x)) \cdot \phi_j(x)$
        * $m_j(n+1) = w_j(n) + \eta \cdot (y - F(x)) \cdot w_j(n) \cdot \phi_j(x) \cdot \frac{1}{{\sigma_j(n)}^2} \cdot |x-m_j(n)|$
        * $\sigma_j(n+1) = w_j(n) + \eta \cdot (y - F(x)) \cdot w_j(n) \cdot \phi_j(x) \cdot \frac{1}{{\sigma_j(n)}^3} \cdot |x-m_j(n)|^2$
4. 直到 $epoch$ 完成，開始實際測試並畫圖，但車子還是有可能撞牆🥲，抵達終點或撞牆會停止
5. 將沿途結果輸出成 `track?D.txt` 與 `result.gif`，`result.gif` 範例如下

![](https://i.imgur.com/Xtn4wW2.gif)

---

## 訓練集資料
* `train4dAll.txt` 每行分別是「前方距離」、「右前方距離」、「左前方距離」與「方向盤預期調整角度」
* `train6dAll.txt` 每行分別是「x 座標」、「y 座標」、「前方距離」、「右前方距離」、「左前方距離」與「方向盤預期調整角度」
* 方向盤調整角度向右為正，向左為負
* `result.gif` 中，綠色直線代表三個感測器
* 其實應該是 3D 跟 5D 吧🤔
