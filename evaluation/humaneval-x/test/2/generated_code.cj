//from std import math.*
main() {
    try{
    let correct = ArrayList<Bool>(
        truncateNumber(3.5) == 0.5,
        abs(truncateNumber(1.33) - 0.33) < 1e-6,
        abs(truncateNumber(123.456) - 0.456) < 1e-6
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