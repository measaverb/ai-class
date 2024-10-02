import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 (이미지를 3채널로 변환, VGG-16은 RGB 3채널 입력을 요구)
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))

# VGG-16 모델 로드 (ImageNet으로 사전 학습된 가중치 사용, 최상위 레이어는 제거)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(28, 28, 3))

# 기존 가중치를 고정 (최상위 레이어 제외)
base_model.trainable = False

# VGG-16 위에 새로운 레이어 추가
model = models.Sequential(
    [
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),  # MNIST는 10개의 클래스로 분류
    ]
)

# 모델 컴파일
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 모델 요약 출력
model.summary()

# 모델 훈련
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 마지막 몇 개 층만 학습하도록 설정 (Fine-tuning)
# VGG-16의 마지막 몇 개의 층을 학습 가능하도록 설정
base_model.trainable = True
fine_tune_at = 15  # 마지막 15번째 층부터는 학습

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 모델 재컴파일 (학습 가능한 가중치 업데이트)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 작은 학습률로 설정
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 훈련 (Fine-tuning)
history_fine = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 테스트 데이터로 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

# 정확도 그래프 출력
plt.plot(history.history["accuracy"], label="Train Accuracy (Initial)")
plt.plot(history.history["val_accuracy"], label="Test Accuracy (Initial)")
plt.plot(history_fine.history["accuracy"], label="Train Accuracy (Fine-tuned)")
plt.plot(history_fine.history["val_accuracy"], label="Test Accuracy (Fine-tuned)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
