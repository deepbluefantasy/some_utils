def compute_probability(logits, temperature, top_k):
    print(f"logits={logits}, temperature={temperature}, top_k={top_k}\n")
    # logits的exp总和，即softmax的分母
    total = 0
    # logits温度变换后的exp总和，即softmax的分母
    total_t = 0
    for i, logit in enumerate(logits):
        total += exp(logit)
        total_t += exp(logit/temperature)

    print(f"【未使用top-k之前的概率】")
    probabilities, probabilities_t = [], []
    for logit in logits:
        probabilities.append(exp(logit)/total*100)
        probabilities_t.append(exp(logit/temperature)/total_t*100)
        print(f"logit为{logit}的token概率为{probabilities[-1]:.2f}%\n经过温度变换后，其logit为{logit/temperature}")
        print(f"变换后的概率为{probabilities_t[-1]:.2f}%")
        print()

    print(f"【使用了top-k之后的概率】")
    for i, logit in enumerate(logits):
        print(f"top-{i+1}的原始概率为{probabilities[i]:.2f}%, top-k的概率和为{sum(probabilities[:top_k]):.2f}%, 因此top-k归一化之后的概率为{probabilities[i]/sum(probabilities[:top_k])*100:.2f}%")
        print(f"top-{i+1}的温度变换后概率为{probabilities_t[i]:.2f}%, top-k的概率和为{sum(probabilities_t[:top_k]):.2f}%, 因此top-k归一化之后的概率为{probabilities_t[i]/sum(probabilities_t[:top_k])*100:.2f}%")
        print()
