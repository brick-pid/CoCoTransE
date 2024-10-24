main() {
    try{
    let correct = ArrayList<Bool>(
            rightAngleTriangle(3, 4, 5) == true,
            rightAngleTriangle(1, 2, 3) == false,
            rightAngleTriangle(10, 6, 8) == true,
            rightAngleTriangle(2, 2, 2) == false,
            rightAngleTriangle(7, 24, 25) == true,
            rightAngleTriangle(10, 5, 7) == false,
            rightAngleTriangle(5, 12, 13) == true,
            rightAngleTriangle(15, 8, 17) == true,
            rightAngleTriangle(48, 55, 73) == true,
            rightAngleTriangle(1, 1, 1) == false,
            rightAngleTriangle(2, 2, 10) == false
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