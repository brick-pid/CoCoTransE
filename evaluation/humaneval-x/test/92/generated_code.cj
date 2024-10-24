main() {
    try{
    let correct = ArrayList<Bool>(
            anyInt(2, 3, 1) == true,
            anyInt(2.5, 2, 3) == false,
            anyInt(1.5, 5, 3.5) == false,
            anyInt(2, 6, 2) == false,
            anyInt(4, 2, 2) == true,
            anyInt(2.2, 2.2, 2.2) == false,
            anyInt(-4, 6, 2) == true,
            anyInt(2, 1, 1) == true,
            anyInt(3, 4, 7) == true,
            anyInt(3.0, 4, 7) == false
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