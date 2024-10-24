main() {
    try{
    let correct = ArrayList<Bool>(
            multiply(148, 412) == 16,
            multiply(19, 28) == 72,
            multiply(2020, 1851) == 0,
            multiply(14,-15) == 20,
            multiply(76, 67) == 42,
            multiply(17, 27) == 49,
            multiply(0, 1) == 0,
            multiply(0, 0) == 0
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}