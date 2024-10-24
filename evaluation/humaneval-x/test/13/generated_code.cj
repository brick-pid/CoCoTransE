main() {
    try{
    let correct = ArrayList<Bool>(
            greatestCommonDivisor(3, 7) == 1,
            greatestCommonDivisor(10, 15) == 5,
            greatestCommonDivisor(49, 14) == 7,
            greatestCommonDivisor(144, 60) == 12
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