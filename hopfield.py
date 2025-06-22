import numpy as np


def create_weights(vector1):
    return np.outer(2 * vector1 - 1, 2 * vector1 - 1) - np.eye(len(vector1))


def sign(val):
    return 1 if val > 0 else 0


def hopfield_network(y0, source_vector, mode='1', max_iters=100, show_info=False):
    max_iters = int(max_iters)
    weights = create_weights(source_vector)
    if show_info:
        print(f"Исходный вектор: {source_vector}")
        print(f"Зашумленный образ: {y0}")
    updated_y = y0.copy()
    epoch = 0
    if mode == '1':
        while True:
            for row_index in range(len(updated_y)):
                S = updated_y.dot(weights[row_index])
                updated_y[row_index] = sign(S)
                epoch += 1
            if epoch > max_iters:
                if show_info:
                    print("Превышено максимальное число итераций")
                break
            if np.array_equal(source_vector, updated_y):
                if show_info:
                    print("Состояние сети стабилизировалось.")
                break
    else:
        while True:
            S = updated_y.dot(weights)
            updated_y = np.array([sign(val) for val in S])
            epoch += 1
            if epoch > max_iters:
                if show_info:
                    print("Превышено максимальное число итераций")
                break
            if np.array_equal(source_vector, updated_y):
                if show_info:
                    print("Состояние сети стабилизировалось.")
                break
    if show_info:
        print("Финальный результат:")
        print(updated_y)
    return updated_y


def add_noise_fixed(vector, num_flips):
    noisy = vector.copy()
    indices = np.random.choice(np.arange(len(vector)), size=num_flips, replace=False)
    for idx in indices:
        noisy[idx] = 1 - noisy[idx]
    return noisy


def test_incremental_hopfield_network(mode, max_iters=100, show_data=False):
    np.random.seed(42)
    vectors = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    ])
    mode_str = "Asynchronous" if mode == '1' else "Synchronous"
    for vector_index, ideal in enumerate(vectors):
        if show_data:
            print(f"\nТестирование для вектора #{vector_index + 1}:")
            print("Идеальный вектор:", ideal)
        table_data = []
        noise_level = 1
        reliability = 100.0
        while noise_level <= len(ideal) and reliability >= 0.5:
            trials = 10
            successes = 0
            sample_noisy = None
            sample_recognized = None
            sample_h_dist = None
            for trial in range(trials):
                noisy = add_noise_fixed(ideal, noise_level)
                h_dist = np.sum(ideal != noisy)
                recognized = hopfield_network(noisy, ideal, mode=mode, max_iters=max_iters, show_info=False)
                if np.array_equal(recognized, ideal):
                    successes += 1
                if trial == 0:
                    sample_noisy = noisy.copy()
                    sample_recognized = recognized.copy()
                    sample_h_dist = int(h_dist)
            avg_success = successes / trials
            reliability = avg_success
            if show_data:
                print(f"Инвертированных бит: {noise_level:2d} | Средняя успешность: {avg_success:.2f}")
            table_data.append({
                "Inv": noise_level,
                "Orig": ''.join(map(str, ideal)),
                "Trl": trials,
                "Noisy": ''.join(map(str, sample_noisy)),
                "Rec": ''.join(map(str, sample_recognized)),
                "Ham": sample_h_dist,
                "Rel": avg_success * 100.0,
                "Mode": mode_str
            })
            if avg_success >= 0.5:
                noise_level += 1
            else:
                extra_noise = noise_level + 1
                if extra_noise <= len(ideal):
                    extra_successes = 0
                    for trial in range(trials):
                        noisy = add_noise_fixed(ideal, extra_noise)
                        recognized = hopfield_network(noisy, ideal, mode=mode, max_iters=max_iters, show_info=False)
                        if np.array_equal(recognized, ideal):
                            extra_successes += 1
                    extra_avg = extra_successes / trials
                    if show_data:
                        print(f"Extra test for {extra_noise} flipped bits | Average success: {extra_avg:.2f}")
                break
        if not (avg_success < 0.5):
            if show_data:
                print("Hopfield network restored the pattern even with maximum noise level.")
        print(f"\nТаблица результатов для вектора №{vector_index + 1}:")
        header = "|{:<3}|{:<21}|{:<3}|{:<21}|{:<21}|{:<3}|{:<7}|{:<10}|".format(
            "Inv", "Orig", "Trl", "Noisy", "Rec", "Ham", "Rel(%)", "Mode"
        )
        print(header)
        print("-" * len(header))
        for row in table_data:
            print("|{:<3}|{:<21}|{:<3}|{:<21}|{:<21}|{:<3}|{:<7.2f}|{:<10}|".format(
                row["Inv"],
                row["Orig"],
                row["Trl"],
                row["Noisy"],
                row["Rec"],
                row["Ham"],
                row["Rel"],
                row["Mode"]
            ))
        print("-" * len(header))