import numpy as np


def preliminary_layer(input_pattern, patterns, T=None):
    x = np.array(input_pattern)
    n = len(x)
    if T is None:
        T = n / 2
    activations = []
    for pattern in patterns:
        p = np.array(pattern)
        dot_prod = np.dot(x, p)
        yj = 0.5 * dot_prod + T
        activations.append(yj)
    return np.array(activations)


def relax(z_init, e, max_iters=100):
    z = z_init.copy()
    for t in range(max_iters):
        total = np.sum(z)
        S = (1 + e) * z - e * total
        z_new = np.where(S > 0, S, 0)
        if np.array_equal(z_new, z):
            break
        z = z_new
        if np.count_nonzero(z) == 1:
            break
    return z


def hamming_network(input_vector, patterns, e=0.24, max_iters=50, T=None, show_second_layer=False):
    initial = preliminary_layer(input_vector, patterns, T)
    if show_second_layer:
        print("Выход первого слоя (активации):", initial)
    relaxed = relax(initial, e, max_iters)
    if show_second_layer:
        print("Выход второго слоя (конкурентная релаксация):", relaxed)

    active_indices = np.nonzero(relaxed)[0]
    recognized_index = active_indices[0] if len(active_indices) == 1 else int(np.argmax(relaxed))
    recognized = patterns[recognized_index]
    return initial, recognized


def hamming_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны быть одинаковой длины.")
    return np.sum(vector1 != vector2)


def add_noise_fixed(vector, num_flips):
    noisy = vector.copy()
    indices = np.random.choice(np.arange(len(vector)), size=num_flips, replace=False)
    for idx in indices:
        noisy[idx] = 1 - noisy[idx]
    return noisy


def test_incremental_hamming_network(e=0.24, max_iters=50, T=None, show_second_layer=False, show_info=False):
    np.random.seed(42)
    patterns = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    ])

    if show_info:
        print("\nTesting Hamming Network on each target pattern with incremental noise:")

    num_targets = len(patterns)
    for sample_index in range(num_targets):
        ideal = patterns[sample_index]
        if show_info:
            print(f"\nТестирование для целевого вектора {sample_index + 1}: {''.join(map(str, ideal))}")
        table_data = []
        noise_level = 1
        threshold_found = False

        while noise_level <= len(ideal):
            successes = 0
            sample_noisy = None
            sample_recognized = None
            sample_hamming = None

            for trial in range(10):
                noisy = add_noise_fixed(ideal, noise_level)
                h_dist = hamming_distance(ideal, noisy)
                y, recognized = hamming_network(noisy, patterns, e, max_iters, T, show_second_layer)
                if np.array_equal(recognized, ideal):
                    successes += 1
                if trial == 0:
                    sample_noisy = noisy.copy()
                    sample_recognized = recognized.copy()
                    sample_hamming = int(h_dist)

            avg_success = successes / 10.0
            reliability = avg_success * 100.0
            if show_info:
                print(f"Перевернутых бит: {noise_level:2d} | Средняя успешность распознавания: {avg_success:.2f}")

            table_data.append({
                "Test": noise_level,
                "Orig": ideal.copy(),
                "Trl": 10,
                "Noisy": sample_noisy,
                "Rec": sample_recognized,
                "Ham": sample_hamming,
                "Rel": reliability
            })

            if avg_success >= 0.5:
                noise_level += 1
            else:
                extra_noise = noise_level + 1
                if extra_noise <= len(ideal):
                    extra_success = 0
                    for trial in range(10):
                        noisy = add_noise_fixed(ideal, extra_noise)
                        y, recognized = hamming_network(noisy, patterns, e, max_iters, T, show_second_layer)
                        if np.array_equal(recognized, ideal):
                            extra_success += 1
                    extra_avg = extra_success / 10.0
                    if show_info:
                        print(
                            f"Дополнительное тестирование для {extra_noise} перевернутых бит | Средняя успешность: {extra_avg:.2f}")
                threshold_found = True
                break

        if not threshold_found:
            if show_info:
                print("Hamming сеть распознала образ даже при максимальном уровне шума.")

        print(f"\nТаблица результатов тестов для целевого вектора №{sample_index + 1}:")
        header = "|{:<4}|{:<18}|{:<4}|{:<18}|{:<18}|{:<4}|{:<8}|".format("Test", "Orig", "Trl", "Noisy", "Rec", "Ham",
                                                                         "Rel(%)")
        print(header)
        print("-" * len(header))
        for row in table_data:
            orig_str = ''.join(map(str, row["Orig"]))
            noisy_str = ''.join(map(str, row["Noisy"]))
            rec_str = ''.join(map(str, row["Rec"]))
            print("|{:<4}|{:<18}|{:<4}|{:<18}|{:<18}|{:<4}|{:<8.2f}|".format(
                row["Test"], orig_str, row["Trl"], noisy_str, rec_str, row["Ham"], row["Rel"]))
        print("-" * len(header))
