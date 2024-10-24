main() {
    try{
    let correct = ArrayList<Bool>(
            bf("Jupiter", "Neptune")==ArrayList<String>("Saturn", "Uranus"),
            bf("Earth", "Mercury")==ArrayList<String>("Venus"),
            bf("Mercury", "Uranus")==ArrayList<String>("Venus", "Earth", "Mars", "Jupiter", "Saturn"),
            bf("Neptune", "Venus")==ArrayList<String>("Earth", "Mars", "Jupiter", "Saturn", "Uranus"),
            bf("Earth", "Earth")==ArrayList<String>(),
            bf("Mars", "Earth")==ArrayList<String>(),
            bf("Jupiter", "Makemake")==ArrayList<String>()
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