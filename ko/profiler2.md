# 프로파일 러를 사용하여 TensorFlow 성능 최적화

[목차]

Profiler와 함께 제공되는 도구를 사용하여 TensorFlow 모델의 성능을 추적합니다. 모델이 호스트 (CPU), 장치 (GPU) 또는 호스트와 장치의 조합에서 어떻게 작동하는지 확인합니다.

프로파일 링은 모델에서 다양한 TensorFlow 작업 (ops)의 하드웨어 리소스 소비 (시간 및 메모리)를 이해하고 성능 병목 현상을 해결하고 궁극적으로 모델을 더 빠르게 실행하는 데 도움이됩니다.

이 가이드는 프로파일 러를 설치하는 방법, 사용 가능한 다양한 도구, 프로파일 러가 성능 데이터를 수집하는 다양한 모드 및 모델 성능을 최적화하기위한 몇 가지 권장 모범 사례를 안내합니다.

Cloud TPU에서 모델 성능을 프로파일 링하려면 [Cloud TPU 가이드를](https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_profile) 참조하세요.

## 프로파일 러 및 GPU 필수 구성 요소 설치

Install the Profiler by downloading and running the [`install_and_run.py`](https://raw.githubusercontent.com/tensorflow/profiler/master/install_and_run.py) script from the [GitHub repository](https://github.com/tensorflow/profiler).

GPU에서 프로파일 링하려면 다음을 수행해야합니다.

1. [Install CUDA® Toolkit 10.1](https://www.tensorflow.org/install/gpu#linux_setup) or newer. CUDA® Toolkit 10.1 supports only single GPU profiling. To profile multiple GPUs, see [Profile multiple GPUs](#profile_multiple_gpus). Ensure that the CUDA® driver version you install is at least 440.33 for Linux or 441.22 for Windows.

2. 경로에 CUPTI가 있는지 확인하십시오.

    ```shell
    /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | \
    grep libcupti
    ```

경로에 CUPTI가 없으면 다음을 실행하여 설치 디렉토리를 `$LD_LIBRARY_PATH` 환경 변수 앞에 추가합니다.

```shell
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

위의 `ldconfig` 명령을 다시 실행하여 CUPTI 라이브러리가 있는지 확인합니다.

### 여러 GPU 프로파일 링 {: id = 'profile_multiple_gpus'}

TensorFlow는 아직 공식적으로 다중 GPU 프로파일 링을 지원하지 않습니다. CUDA® Toolkit 10.2 이상을 설치하여 여러 GPU를 프로파일 링 할 수 있습니다. TensorFlow는 최대 10.1의 CUDA® Toolkit 버전 만 지원하므로 `libcudart.so.10.1` 및 `libcupti.so.10.1` 심볼릭 링크를 만듭니다.

```shell
sudo ln -s /usr/local/cuda/lib64/libcudart.so.10.2 /usr/local/cuda/lib64/libcudart.so.10.1
sudo ln -s /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.2 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.10.1
```

다중 작업자 GPU 구성을 프로파일 링하려면 개별 작업자를 독립적으로 프로파일 링하십시오.

## 프로파일 러 도구

일부 모델 데이터를 캡처 한 후에 만 나타나는 TensorBoard의 **프로필** 탭에서 프로필러에 액세스합니다. Profiler에는 성능 분석에 도움이되는 다양한 도구가 있습니다.

- 개요 페이지
- 입력 파이프 라인 분석기
- TensorFlow 통계
- 추적 뷰어
- GPU 커널 통계

### 개요 페이지

개요 페이지는 프로필 실행 중 모델의 성능에 대한 최상위보기를 제공합니다. 이 페이지에는 호스트 및 모든 장치에 대한 집계 된 개요 페이지와 모델 학습 성능을 개선하기위한 몇 가지 권장 사항이 표시됩니다. 호스트 드롭 다운에서 개별 호스트를 선택할 수도 있습니다.

개요 페이지에는 다음과 같은 데이터가 표시됩니다.

![영상](./images/tf_profiler/overview_page.png)

- **성능 요약-** 모델 성능에 대한 높은 수준의 요약을 표시합니다. 성능 요약은 두 부분으로 구성됩니다.

    1. 단계 시간 분석-평균 단계 시간을 시간이 소비되는 여러 범주로 분류합니다.

        - 컴파일-커널 컴파일에 소요 된 시간
        - 입력-입력 데이터를 읽는 데 소요 된 시간
        - 출력-출력 데이터를 읽는 데 소요 된 시간
        - 커널 시작-호스트가 커널을 시작하는 데 소요 한 시간
        - 호스트 컴퓨팅 시간
        - 장치 대 장치 통신 시간
        - 온 디바이스 컴퓨팅 시간
        - Python 오버 헤드를 포함한 기타 모든 항목

    2. 장치 계산 정밀도-16 비트 및 32 비트 계산을 사용하는 장치 계산 시간의 백분율을보고합니다.

- **단계 시간 그래프-** 샘플링 된 모든 단계에 대한 장치 단계 시간 (밀리 초) 그래프를 표시합니다. 각 단계는 시간이 소요되는 여러 범주 (다른 색상)로 구분됩니다. 빨간색 영역은 장치가 호스트의 입력 데이터를 기다리는 동안 유휴 상태에있는 단계 시간 부분에 해당합니다. 녹색 영역은 장치가 실제로 작동 한 시간을 보여줍니다.

- **기기에서 상위 10 개의 TensorFlow 작업-** 가장 오래 실행 된 기기 내 작업을 표시합니다.

    각 행에는 작업의 자체 시간 (모든 작업에 소요 된 시간의 백분율), 누적 시간, 카테고리 및 이름이 표시됩니다.

- **실행 환경-** 다음을 포함한 모델 실행 환경의 상위 수준 요약을 표시합니다.

    - 사용 된 호스트 수
    - 장치 유형 (GPU / TPU)
    - 장치 코어 수

- **다음 단계에 대한 권장 사항-** 모델이 입력 제한 일 때보고하고 모델 성능 병목 현상을 찾아 해결하는 데 사용할 수있는 도구를 권장합니다.

### 입력 파이프 라인 분석기

TensorFlow 프로그램이 파일에서 데이터를 읽을 때 파이프 라인 방식으로 TensorFlow 그래프의 상단에서 시작됩니다. 읽기 프로세스는 직렬로 연결된 여러 데이터 처리 단계로 나뉩니다. 여기서 한 단계의 출력은 다음 단계의 입력입니다. 이 데이터 읽기 시스템을 *입력 파이프 라인* 이라고합니다.

파일에서 레코드를 읽기위한 일반적인 파이프 라인에는 다음 단계가 있습니다.

1. 파일 읽기
2. 파일 전처리 (선택 사항)
3. 호스트에서 장치로 파일 전송

비효율적 인 입력 파이프 라인은 애플리케이션 속도를 크게 저하시킬 수 있습니다. 애플리케이션은 입력 파이프 라인에서 상당한 시간을 소비 할 때 **입력 바운드** 로 간주됩니다. 입력 파이프 라인 분석기에서 얻은 통찰력을 사용하여 입력 파이프 라인이 비효율적 인 부분을 이해합니다.

입력 파이프 라인 분석기는 프로그램이 입력 바운드 여부를 즉시 알려주고 입력 파이프 라인의 모든 단계에서 성능 병목 현상을 디버그하기 위해 장치 및 호스트 측 분석을 안내합니다.

데이터 입력 파이프 라인을 최적화하기위한 권장 모범 사례는 입력 파이프 라인 성능에 대한 지침을 참조하세요.

#### 입력 파이프 라인 대시 보드

입력 파이프 라인 분석기를 열려면 **프로필** 을 선택한 다음 **도구** 드롭 다운에서 **input_pipeline_analyzer** 를 선택합니다.

![영상](./images/tf_profiler/input_pipeline_analyzer.png)

대시 보드에는 세 개의 섹션이 있습니다.

1. **요약-** 애플리케이션이 입력 바인딩되었는지 여부에 대한 정보와 함께 전체 입력 파이프 라인을 요약합니다.
2. **장치 측 분석-** 장치 단계 시간 및 각 단계에서 코어에서 입력 데이터를 기다리는 데 소요 된 장치 시간 범위를 포함하여 자세한 장치 측 분석 결과를 표시합니다.
3. **호스트 측 분석-** 호스트의 입력 처리 시간 분석을 포함하여 호스트 측에 대한 자세한 분석을 표시합니다.

#### 입력 파이프 라인 요약

요약은 호스트의 입력을 기다리는 데 소요 된 장치 시간의 백분율을 표시하여 프로그램이 입력 바인딩되었는지 여부를보고합니다. 계측 된 표준 입력 파이프 라인을 사용하는 경우 도구는 대부분의 입력 처리 시간이 소요 된 위치를보고합니다.

#### 장치 측 분석

장치 측 분석은 장치와 호스트에서 소비 한 시간 및 호스트에서 입력 데이터를 기다리는 데 소비 한 장치 시간에 대한 통찰력을 제공합니다.

1. **단계 번호에 대해 플로팅 된 단계 시간-** 샘플링 된 모든 단계에 대한 장치 단계 시간 (밀리 초) 그래프를 표시합니다. 각 단계는 시간이 소요되는 여러 범주 (다른 색상)로 구분됩니다. 빨간색 영역은 장치가 호스트의 입력 데이터를 기다리는 동안 유휴 상태에있는 단계 시간 부분에 해당합니다. 녹색 영역은 장치가 실제로 작동 한 시간을 보여줍니다.
2. **단계 시간 통계-장치 단계 시간** 의 평균, 표준 편차 및 범위 ([최소, 최대])를보고합니다.

#### 호스트 측 분석

호스트 측 분석은 호스트의 입력 처리 시간 ( `tf.data` API 작업에 소요 된 시간)을 여러 범주로 분류하여보고합니다.

- **요청시 파일에서 데이터 읽기-** 캐싱, 프리 페치 및 인터리빙없이 파일에서 데이터를 읽는 데 소요 된 시간입니다.
- **미리 파일에서 데이터 읽기-** 캐싱, 프리 페치 및 인터리빙을 포함하여 파일을 읽는 데 소요 된 시간
- **데이터 사전 처리-** 이미지 압축 해제와 같은 사전 처리 작업에 소요 된 시간
- **장치로 전송할** 데이터를 대기열에 추가-데이터를 장치로 전송하기 전에 데이터를 인피 드 대기열에 넣는 데 소요 된 시간

Expand the **Input Op Statistics** to see the statistics for individual input ops and their categories broken down by execution time.

![영상](./images/tf_profiler/input_op_stats.png)

다음 정보가 포함 된 각 항목과 함께 소스 데이터 테이블이 나타납니다.

1. **입력 작업-입력 작업** 의 TensorFlow 작업 이름을 표시합니다.
2. **개수-** 프로파일 링 기간 동안 작업 실행의 총 인스턴스 수를 표시합니다.
3. **총 시간 (밀리 초)-** 각 인스턴스에 소요 된 시간의 누적 합계를 표시합니다.
4. **총 시간 %-작업에** 소요 된 총 시간을 입력 처리에 소요 된 총 시간의 일부로 표시합니다.
5. **총 자체 시간 (ms)-** 각 인스턴스에 소요 된 자체 시간의 누적 합계를 표시합니다. 여기에서 자체 시간은 호출하는 함수에 소요 된 시간을 제외하고 함수 본문 내부에서 소요 된 시간을 측정합니다.
6. **총 자체 시간 %** . 입력 처리에 소요 된 총 시간의 일부로 총 자체 시간을 표시합니다.
7. **카테고리** . 입력 작업의 처리 범주를 표시합니다.

### TensorFlow 통계

TensorFlow Stats 도구는 프로파일 링 세션 중에 호스트 또는 기기에서 실행되는 모든 TensorFlow 작업 (op)의 성능을 표시합니다.

![영상](./images/tf_profiler/tf_stats.png)

이 도구는 두 개의 창에 성능 정보를 표시합니다.

- 상단 창에는 최대 4 개의 원형 차트가 표시됩니다.

    1. 호스트에서 각 작업의 자체 실행 시간 분포
    2. 호스트에서 각 작업 유형의 자체 실행 시간 분포
    3. 장치에서 각 작업의 자체 실행 시간 분포
    4. 장치에서 각 작업 유형의 자체 실행 시간 분포

- 아래쪽 창에는 TensorFlow 작업에 대한 데이터를보고하는 테이블이 표시되며 각 작업에 대해 한 행, 각 데이터 유형에 대해 하나의 열이 있습니다 (열 제목을 클릭하여 열 정렬). 이 테이블의 데이터를 CSV 파일로 내보내려면 상단 창의 오른쪽에있는 CSV로 내보내기 버튼을 클릭합니다.

    참고 :

    - 작업에 하위 작업이있는 경우 :

        - 작업의 총 "누적"시간에는 하위 작업 내부에서 보낸 시간이 포함됩니다.
        - 작업의 총 "자체"시간에는 하위 작업 내에서 보낸 시간이 포함되지 않습니다.

    - 작업이 호스트에서 실행되는 경우 :

        - 작업으로 인해 발생하는 장치의 총 자체 시간 비율은 0입니다.
        - 이 작업을 포함하여 장치에서 총 자체 시간의 누적 백분율은 0이됩니다.

    - 작업이 기기에서 실행되는 경우 :

        - 이 작업으로 인해 호스트에서 발생한 총 자체 시간의 백분율은 0이됩니다.
        - 이 작업을 포함하여 호스트에서 총 자체 시간의 누적 백분율은 0이됩니다.

원형 차트 및 표에 유휴 시간을 포함하거나 제외하도록 선택할 수 있습니다.

### 추적 뷰어

추적 뷰어는 다음을 보여주는 타임 라인을 표시합니다.

- TensorFlow 모델에 의해 실행 된 작업의 기간
- 시스템의 어느 부분 (호스트 또는 장치)이 작업을 실행했는지. 일반적으로 호스트는 입력 작업을 실행하고 훈련 데이터를 전처리하여 장치로 전송하는 반면 장치는 실제 모델 훈련을 실행합니다.

추적 뷰어를 사용하면 모델의 성능 문제를 식별 한 다음 해결 단계를 수행 할 수 있습니다. 예를 들어 높은 수준에서 입력 또는 모델 학습이 대부분의 시간을 차지하는지 여부를 식별 할 수 있습니다. 드릴 다운하면 실행하는 데 가장 오래 걸리는 작업을 식별 할 수 있습니다.

트레이스 뷰어는 장치 당 1 백만 개의 이벤트로 제한됩니다.

#### 추적 뷰어 인터페이스

트레이스 뷰어를 열면 가장 최근 실행이 표시됩니다.

![영상](./images/tf_profiler/trace_viewer.png)

이 화면에는 다음과 같은 주요 요소가 있습니다.

1. **타임 라인 창-** 시간이 지남에 따라 기기와 호스트가 실행 한 작업을 표시합니다.
2. **세부 정보 창-** 타임 라인 창에서 선택한 작업에 대한 추가 정보를 표시합니다.

타임 라인 창에는 다음 요소가 포함됩니다.

1. **상단 바-** 다양한 보조 컨트롤 포함
2. **시간 축-** 트레이스 시작을 기준으로 시간을 표시합니다.
3. **섹션 및 트랙 레이블-** 각 섹션에는 여러 트랙이 포함되어 있으며 왼쪽에 삼각형이있어 섹션을 확장 및 축소 할 수 있습니다. 시스템의 모든 처리 요소에 대해 하나의 섹션이 있습니다.
4. **도구 선택기-** 확대 / 축소, 팬, 선택 및 타이밍과 같은 추적 뷰어와 상호 작용하기위한 다양한 도구가 포함되어 있습니다. 시간 간격을 표시하려면 타이밍 도구를 사용하십시오.
5. **이벤트-작업** 이 실행 된 시간 또는 학습 단계와 같은 메타 이벤트 기간을 표시합니다.

##### 섹션 및 트랙

추적 뷰어에는 다음 섹션이 포함되어 있습니다.

- **각 장치 노드에 대해 하나의 섹션** 으로, 장치 칩 번호와 칩 내의 장치 노드 (예 `/device:GPU:0 (pid 0)` )로 레이블이 지정됩니다. 각 장치 노드 섹션에는 다음 트랙이 포함됩니다.
    - **단계-** 장치에서 실행 된 훈련 단계의 기간을 표시합니다.
    - **TensorFlow Ops-** . 기기에서 실행 된 작업을 표시합니다.
    - **XLA** 작업-XLA가 사용 된 컴파일러 인 경우 기기에서 실행 된 [XLA](https://www.tensorflow.org/xla/) 작업 (작업)을 표시합니다 (각 TensorFlow 작업은 하나 또는 여러 개의 XLA 작업으로 변환됩니다. XLA 컴파일러는 XLA 작업을 기기에서 실행되는 코드로 변환합니다).
- **호스트 시스템의 CPU에서 실행되는 스레드에 대한 한 섹션 인** **"Host Threads"** . 섹션에는 각 CPU 스레드에 대한 하나의 트랙이 포함됩니다. 참고 : 섹션 레이블과 함께 표시된 정보는 무시할 수 있습니다.

##### 이벤트

타임 라인 내의 이벤트는 다른 색상으로 표시됩니다. 색상 자체에는 특별한 의미가 없습니다.

### GPU 커널 통계

이 도구는 모든 GPU 가속 커널에 대한 성능 통계 및 원래 작업을 보여줍니다.

![영상](./images/tf_profiler/gpu_kernel_stats.png)

이 도구는 두 개의 창에 정보를 표시합니다.

- 상단 창에는 총 경과 시간이 가장 높은 CUDA 커널을 보여주는 원형 차트가 표시됩니다.

- 아래쪽 창에는 고유 한 각 커널 작업 쌍에 대한 다음 데이터가 포함 된 테이블이 표시됩니다.

    - 커널 작업 쌍으로 그룹화 된 총 경과 GPU 기간의 내림차순 순위
    - 시작된 커널의 이름
    - 커널에서 사용하는 GPU 레지스터 수
    - 사용 된 공유 (정적 + 동적 공유) 메모리의 총 크기 (바이트)
    - `blockDim.x, blockDim.y, blockDim.z` 로 표현되는 블록 차원
    - `gridDim.x, gridDim.y, gridDim.z` 로 표현되는 그리드 치수
    - 작업이 TensorCore를 사용할 수 있는지 여부
    - 커널에 TensorCore 명령어가 포함되어 있는지 여부
    - 이 커널을 시작한 작업의 이름
    - 이 커널 작업 쌍의 발생 수
    - 총 경과 GPU 시간 (마이크로 초)
    - 평균 GPU 경과 시간 (마이크로 초)
    - 최소 경과 GPU 시간 (마이크로 초)
    - 마이크로 초 단위의 최대 GPU 경과 시간

## 성능 데이터 수집

TensorFlow Profiler는 TensorFlow 모델의 호스트 활동 및 GPU 추적을 수집합니다. 프로그래밍 모드 또는 샘플링 모드를 통해 성능 데이터를 수집하도록 프로파일 러를 구성 할 수 있습니다.

- TensorBoard Keras 콜백 ( `tf.keras.callbacks.TensorBoard` )을 사용하는 프로그래밍 모드

    ```python
    # Profile from batches 10 to 15
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch='10, 15')

    # Train the model and use the TensorBoard Keras callback to collect
    # performance profiling data
    model.fit(train_data,
              steps_per_epoch=20,
              epochs=5,
              callbacks=[tb_callback])
    ```

- `tf.profiler` 함수 API를 사용하는 프로그래밍 모드

    ```python
    tf.profiler.experimental.start('logdir')
    # Train the model here
    tf.profiler.experimental.stop()
    ```

- 컨텍스트 관리자를 사용하는 프로그래밍 모드

    ```python
    with tf.profiler.experimental.Profile('logdir'):
        # Train the model here
        pass
    ```

참고 : 프로파일 러를 너무 오래 실행하면 메모리가 부족해질 수 있습니다. 한 번에 10 개 이하의 단계를 프로파일 링하는 것이 좋습니다. 초기화 오버 헤드로 인한 부정확성을 피하기 위해 처음 몇 개의 배치를 프로파일 링하지 마십시오.

- 샘플링 모드 `tf.profiler.experimental.server.start()` 를 사용하여 주문형 프로파일 링을 수행하여 TensorFlow 모델 실행으로 gRPC 서버를 시작합니다. gRPC 서버를 시작하고 모델을 실행 한 후 TensorBoard 프로필 플러그인의 **Capture Profile** 버튼을 통해 프로필을 캡처 할 수 있습니다. 위의 Install profiler 섹션의 스크립트를 사용하여 TensorBoard 인스턴스가 아직 실행되고 있지 않은 경우 시작합니다.

    예로서,

    ```python
    # Start a gRPC server at port 6009
    tf.profiler.experimental.server.start(6009)
    # ... TensorFlow program ...
    ```

![영상](./images/tf_profiler/capture_profile.png)

프로파일 서비스 URL 또는 TPU 이름, 프로파일 링 기간, 프로파일 러가 처음에 실패 할 경우 프로파일 캡처를 재 시도 할 횟수를 지정할 수 있습니다.

## 최적의 모델 성능을위한 모범 사례

최적의 성능을 얻으려면 TensorFlow 모델에 해당하는 다음 권장 사항을 사용하세요.

일반적으로 장치에서 모든 변환을 수행하고 플랫폼에 대해 cuDNN 및 인텔 MKL과 같은 최신 호환 라이브러리 버전을 사용하는지 확인하십시오.

### 입력 데이터 파이프 라인 최적화

효율적인 데이터 입력 파이프 라인은 장치 유휴 시간을 줄여 모델 실행 속도를 크게 향상시킬 수 있습니다. 데이터 입력 파이프 라인을보다 효율적으로 만들려면 [여기](https://www.tensorflow.org/guide/data_performance) 에 설명 된대로 다음 모범 사례를 통합하는 것이 좋습니다.

- 데이터 프리 페치
- 데이터 추출 병렬화
- 데이터 변환 병렬화
- 메모리에 데이터 캐시
- 사용자 정의 함수 벡터화
- 변환 적용시 메모리 사용량 감소

또한 합성 데이터로 모델을 실행하여 입력 파이프 라인이 성능 병목인지 확인하십시오.

### 장치 성능 향상

- 훈련 미니 배치 크기 늘리기 (훈련 루프의 한 번 반복에서 장치 당 사용되는 훈련 샘플 수)
- TF 통계를 사용하여 온 디바이스 작업이 얼마나 효율적으로 실행되는지 확인
- `tf.function` 을 사용하여 계산을 수행하고 선택적으로 `experimental_compile` 플래그를 활성화합니다.
- 단계 사이의 호스트 Python 작업을 최소화하고 콜백을 줄입니다. 모든 단계가 아닌 몇 단계마다 메트릭 계산
- 장치 컴퓨팅 장치를 계속 바쁘게 유지
- 병렬로 여러 장치에 데이터 보내기
- 채널을 먼저 선호하도록 데이터 레이아웃을 최적화합니다 (예 : NHWC보다 NCHW). NVIDIA® V100과 같은 특정 GPU는 NHWC 데이터 레이아웃에서 더 나은 성능을 발휘합니다.
- Consider using 16-bit numerical representations such as `fp16`, the half-precision floating point format specified by IEEE or the Brain floating-point [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) format
- [Keras 혼합 정밀도 API](https://www.tensorflow.org/guide/keras/mixed_precision) 사용 고려
- GPU에서 훈련 할 때 TensorCore를 사용하십시오. GPU 커널은 정밀도가 fp16이고 입력 / 출력 차원이 8 또는 16 (int8의 경우)으로 나눌 수있는 경우 TensorCore를 사용합니다.

## 추가 자료

- See the end-to-end [TensorBoard profiler tutorial](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) to implement the advice in this guide.
- Watch the [Performance profiling in TF 2](https://www.youtube.com/watch?v=pXHAQIhhMhI) talk from the TensorFlow Dev Summit 2020.
