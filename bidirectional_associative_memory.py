import numpy as np


def activation_function(s):
    return np.where(s > 0, 1, np.where(s < 0, -1, 0))


def binary_to_bipolar(vector):
    return np.where(vector == 0, -1, 1)


def bipolar_to_binary(vector):
    return np.where(vector == -1, 0, 1)


def create_weights(input_vector, output_vector):
    bipolar_input = binary_to_bipolar(input_vector)
    bipolar_output = binary_to_bipolar(output_vector)
    return np.outer(bipolar_input, bipolar_output)


def train_bam(pairs):
    if not pairs:
        raise ValueError("Нет пар для обучения.")
    weights = None
    for inp, out in pairs:
        bipolar_inp = binary_to_bipolar(inp)
        bipolar_out = binary_to_bipolar(out)
        if weights is None:
            weights = np.outer(bipolar_inp, bipolar_out)
        else:
            weights += np.outer(bipolar_inp, bipolar_out)
    return weights


def bam_recall(input_vector, weights, reverse=False):
    bipolar_input = binary_to_bipolar(input_vector)
    if reverse:
        bipolar_output = activation_function(np.dot(bipolar_input, weights.T))
    else:
        bipolar_output = activation_function(np.dot(bipolar_input, weights))
    return bipolar_output


def add_noise_fixed(vector, num_flips):
    noisy = vector.copy()
    indices = np.random.choice(np.arange(len(vector)), size=num_flips, replace=False)
    for idx in indices:
        noisy[idx] = 1 - noisy[idx]
    return noisy


def test_incremental_bam(show_data=False):
    np.random.seed(42)
    matrix_x = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    ])
    matrix_y = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
    ])

    num_pairs = matrix_x.shape[0]
    for pair_index in range(num_pairs):
        x_vector = matrix_x[pair_index]
        y_vector = matrix_y[pair_index]

        if show_data:
            print(f"\nPair #{pair_index + 1}:")
            print("  X (input):", ''.join(map(str, x_vector)))
            print("  Y (output):", ''.join(map(str, y_vector)))

        for part in [1, 2]:
            if part == 1:
                original = x_vector.copy()
                target = y_vector.copy()
                max_noise = len(original)
                part_str = "X"
                weights = create_weights(original, target)
                predictor = lambda noisy: bam_recall(noisy, weights, reverse=False)
            elif part == 2:
                original = y_vector.copy()
                target = x_vector.copy()
                max_noise = len(original)
                part_str = "Y"
                weights = create_weights(target, original)
                predictor = lambda noisy: bam_recall(noisy, weights, reverse=True)
            else:
                continue

            if show_data:
                print(f"\nTesting BAM on {part_str} with 10 trials per noise level (0 to {max_noise} flipped bits).")

            table_data = []
            noise_level = 0
            threshold_found = False

            while noise_level <= max_noise:
                trials = 10
                successes = 0
                sample_noisy = None
                sample_predicted = None
                sample_h_dist = None

                for trial in range(trials):
                    noisy = add_noise_fixed(original, noise_level)
                    h_dist = np.sum(original != noisy)
                    predicted = predictor(noisy)
                    if np.array_equal(predicted, binary_to_bipolar(target)):
                        successes += 1
                    if trial == 0:
                        sample_noisy = noisy.copy()
                        sample_predicted = predicted.copy()
                        sample_h_dist = int(h_dist)

                avg_success = successes / trials
                if show_data:
                    print(f"Flipped bits in {part_str}: {noise_level:2d} | Average prediction success: {avg_success:.2f}")

                table_data.append({
                    "Flipped Bits": noise_level,
                    "Original Vector": original.copy(),
                    "Trials": trials,
                    "Noisy Vector": sample_noisy.copy(),
                    "Predicted Vector": bipolar_to_binary(sample_predicted),
                    "Hamming Distance": sample_h_dist,
                    "Avg Reliability (%)": avg_success * 100.0,
                    "Part": part_str
                })

                if avg_success >= 0.5:
                    noise_level += 1
                else:
                    extra_noise = noise_level + 1
                    if extra_noise <= max_noise:
                        extra_successes = 0
                        for trial in range(trials):
                            noisy = add_noise_fixed(original, extra_noise)
                            predicted = predictor(noisy)
                            if np.array_equal(predicted, binary_to_bipolar(target)):
                                extra_successes += 1
                        extra_avg = extra_successes / trials
                        if show_data:
                            print(f"Extra test for {extra_noise} flipped bits in {part_str} | Average prediction success: {extra_avg:.2f}")
                    threshold_found = True
                    break

            if not threshold_found:
                if show_data:
                    print(f"\nBAM produced correct predictions even at maximum noise level in {part_str}.")

            print(f"\nТаблица результатов тестирования BAM по части {part_str} для пары №{pair_index + 1}:")
            header = "|{:<12}|{:<18}|{:<6}|{:<18}|{:<18}|{:<6}|{:<10}|{:<6}|".format(
                "Flipped", "Orig", "Trls", "Noisy", "Predicted", "Ham", "Avg Rel(%)", "Part"
            )
            print(header)
            print("-" * len(header))
            for row in table_data:
                orig_str = ''.join(map(str, row["Original Vector"]))
                noisy_str = ''.join(map(str, row["Noisy Vector"]))
                pred_str = ''.join(map(str, row["Predicted Vector"]))
                print("|{:<12}|{:<18}|{:<6}|{:<18}|{:<18}|{:<6}|{:<10.2f}|{:<6}|".format(
                    row["Flipped Bits"],
                    orig_str,
                    row["Trials"],
                    noisy_str,
                    pred_str,
                    row["Hamming Distance"],
                    row["Avg Reliability (%)"],
                    row["Part"]
                ))
            print("-" * len(header))