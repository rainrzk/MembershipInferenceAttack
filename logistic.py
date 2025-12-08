# %%
# Set-up
import os

path = r"./"
if os.path.exists(path):
    print("ok")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import fashion_mnist 

# 1) Load Fashion-MNIST
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

# 2) Flatten and scale to [0,1]
X_train_full = X_train_full.reshape(len(X_train_full), -1) / 255.0
X_test_full  = X_test_full.reshape(len(X_test_full), -1) / 255.0

def membership_attack_accuracy_from_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Attack prediction: 1 if classified correctly (IN), 0 otherwise (OUT)
    train_correct = (y_pred_train == y_train).astype(int)
    test_correct  = (y_pred_test == y_test).astype(int)

    attack_preds  = np.concatenate([train_correct, test_correct])
    attack_labels = np.concatenate([np.ones_like(train_correct),  # true IN for train
                                    np.zeros_like(test_correct)]) # true OUT for test

    return (attack_preds == attack_labels).mean()

def predict_with_params(X, W, b):
    logits = X @ W.T + b      # shape: (n_samples, n_classes)
    return logits.argmax(axis=1)

def membership_attack_accuracy_from_params(W, b, X_train, y_train, X_test, y_test):
    y_pred_train = predict_with_params(X_train, W, b)
    y_pred_test  = predict_with_params(X_test,  W, b)

    train_correct = (y_pred_train == y_train).astype(int)
    test_correct  = (y_pred_test  == y_test ).astype(int)

    attack_preds  = np.concatenate([train_correct, test_correct])
    attack_labels = np.concatenate([np.ones_like(train_correct),
                                    np.zeros_like(test_correct)])
    return (attack_preds == attack_labels).mean()

# Membership Inference Attack

n_list = [100, 200, 400, 800, 1600, 2500, 5000, 10000]

train_acc_unreg = []
test_acc_unreg  = []
train_acc_reg   = []
test_acc_reg    = []

attack_acc_unreg = []
attack_acc_reg   = []

for n in n_list:
    X_train = X_train_full[:n]
    y_train = y_train_full[:n]
    X_test  = X_test_full[:n]
    y_test  = y_test_full[:n]

    # Unregularized
    clf_unreg = LogisticRegression(
        penalty=None,
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        n_jobs=-1
    )
    clf_unreg.fit(X_train, y_train)
    train_acc_unreg.append(clf_unreg.score(X_train, y_train))
    test_acc_unreg.append(clf_unreg.score(X_test,  y_test))

    # L2-regularized
    clf_reg = LogisticRegression(
        penalty='l2',
        C=0.01,
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        n_jobs=-1
    )
    clf_reg.fit(X_train, y_train)
    train_acc_reg.append(clf_reg.score(X_train, y_train))
    test_acc_reg.append(clf_reg.score(X_test,  y_test))

    attack_acc_unreg.append(
        membership_attack_accuracy_from_model(clf_unreg, X_train, y_train, X_test, y_test)
    )
    attack_acc_reg.append(
        membership_attack_accuracy_from_model(clf_reg, X_train, y_train, X_test, y_test)
    )

plt.figure()
plt.plot(n_list, train_acc_unreg, marker='o', label='Train (no reg)')
plt.plot(n_list, test_acc_unreg,  marker='o', label='Test (no reg)')
plt.xlabel("n")
plt.ylabel("Accuracy")
plt.title("Logistic regression (no reg)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(n_list, train_acc_reg, marker='o', label='Train (L2, C=0.01)')
plt.plot(n_list, test_acc_reg,  marker='o', label='Test (L2, C=0.01)')
plt.xlabel("n")
plt.ylabel("Accuracy")
plt.title("Logistic regression (L2, C=0.01)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(n_list, attack_acc_unreg, marker='o', label='Attack (no reg)')
plt.plot(n_list, attack_acc_reg,   marker='o', label='Attack (L2, C=0.01)')
plt.xlabel("n")
plt.ylabel("Membership inference accuracy")
plt.title("Membership inference attack")
plt.legend()
plt.grid(True)
plt.show()

# Membership Inference Defense

n = 400
X_train = X_train_full[:n]
y_train = y_train_full[:n]
X_test  = X_test_full[:n]
y_test  = y_test_full[:n]

base_unreg = LogisticRegression(
    penalty=None,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)
base_unreg.fit(X_train, y_train)

base_reg = LogisticRegression(
    penalty='l2',
    C=0.1,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    n_jobs=-1
)
base_reg.fit(X_train, y_train)

W_unreg = base_unreg.coef_.copy()
b_unreg = base_unreg.intercept_.copy()
W_reg   = base_reg.coef_.copy()
b_reg   = base_reg.intercept_.copy()

sigma2_values = np.linspace(0.0, 5.0, 11)
num_trials = 20

train_acc_unreg_noise = []
test_acc_unreg_noise  = []
attack_unreg_noise    = []

train_acc_reg_noise = []
test_acc_reg_noise  = []
attack_reg_noise    = []

for sigma2 in sigma2_values:
    sigma = np.sqrt(sigma2)

    tr_acc_u, te_acc_u, att_u = [], [], []
    tr_acc_r, te_acc_r, att_r = [], [], []

    for _ in range(num_trials):
        noise_W_u = np.random.normal(0.0, sigma, size=W_unreg.shape)
        noise_b_u = np.random.normal(0.0, sigma, size=b_unreg.shape)
        W_u = W_unreg + noise_W_u
        b_u = b_unreg + noise_b_u

        y_train_pred_u = predict_with_params(X_train, W_u, b_u)
        y_test_pred_u  = predict_with_params(X_test,  W_u, b_u)
        tr_acc_u.append((y_train_pred_u == y_train).mean())
        te_acc_u.append((y_test_pred_u  == y_test ).mean())
        att_u.append(
            membership_attack_accuracy_from_params(W_u, b_u, X_train, y_train, X_test, y_test)
        )

        noise_W_r = np.random.normal(0.0, sigma, size=W_reg.shape)
        noise_b_r = np.random.normal(0.0, sigma, size=b_reg.shape)
        W_r = W_reg + noise_W_r
        b_r = b_reg + noise_b_r

        y_train_pred_r = predict_with_params(X_train, W_r, b_r)
        y_test_pred_r  = predict_with_params(X_test,  W_r, b_r)
        tr_acc_r.append((y_train_pred_r == y_train).mean())
        te_acc_r.append((y_test_pred_r  == y_test ).mean())
        att_r.append(
            membership_attack_accuracy_from_params(W_r, b_r, X_train, y_train, X_test, y_test)
        )

    train_acc_unreg_noise.append(np.mean(tr_acc_u))
    test_acc_unreg_noise.append(np.mean(te_acc_u))
    attack_unreg_noise.append(np.mean(att_u))

    train_acc_reg_noise.append(np.mean(tr_acc_r))
    test_acc_reg_noise.append(np.mean(te_acc_r))
    attack_reg_noise.append(np.mean(att_r))

plt.figure()
plt.plot(sigma2_values, train_acc_unreg_noise, marker='o', label='Train (no reg)')
plt.plot(sigma2_values, test_acc_unreg_noise,  marker='o', label='Test (no reg)')
plt.plot(sigma2_values, train_acc_reg_noise,   marker='o', label='Train (L2, C=0.1)')
plt.plot(sigma2_values, test_acc_reg_noise,    marker='o', label='Test (L2, C=0.1)')
plt.xlabel(r"$\sigma^2$")
plt.ylabel("Accuracy")
plt.title("Effect of noise on classification accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(sigma2_values, attack_unreg_noise, marker='o', label='Attack (no reg)')
plt.plot(sigma2_values, attack_reg_noise,   marker='o', label='Attack (L2, C=0.1)')
plt.xlabel(r"$\sigma^2$")
plt.ylabel("Membership inference accuracy")
plt.title("Effect of noise on membership inference")
plt.legend()
plt.grid(True)
plt.show()
