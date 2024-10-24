main() {
    try{
    let correct = ArrayList<Bool>(
            isEqualToSumEven(4) == false,
            isEqualToSumEven(6) == false,
            isEqualToSumEven(8) == true,
            isEqualToSumEven(10) == true,
            isEqualToSumEven(11) == false,
            isEqualToSumEven(12) == true,
            isEqualToSumEven(13) == false,
            isEqualToSumEven(16) == true
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