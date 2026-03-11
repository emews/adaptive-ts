from adaptive_ts.example_simulator.odin import generate_ground_truth

def test():
    res = generate_ground_truth()
    return res
if __name__ == "__main__":
    res = test()
    print(res)