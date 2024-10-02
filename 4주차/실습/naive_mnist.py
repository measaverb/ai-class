import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화 (0~255 값을 0~1 사이로 변환)
x_train, x_test = x_train / 255.0, x_test / 255.0

# CNN 모델 정의
model = models.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# 모델 컴파일
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 모델 훈련
history = model.fit(
    x_train[..., tf.newaxis],
    y_train,
    epochs=5,
    validation_data=(x_test[..., tf.newaxis], y_test),
)

# 테스트 데이터로 모델 평가
test_loss, test_acc = model.evaluate(x_test[..., tf.newaxis], y_test, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

# 정확도 그래프 출력
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
