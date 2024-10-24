main() {
    try{
    let correct = ArrayList<Bool>(
            carRaceCollision(2) == 4,
            carRaceCollision(3) == 9,
            carRaceCollision(4) == 16,
            carRaceCollision(8) == 64,
            carRaceCollision(10) == 100
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