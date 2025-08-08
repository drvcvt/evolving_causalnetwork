# Legt die Mindestversion von CMake fest. Verhindert Probleme mit alten Installationen.
cmake_minimum_required(VERSION 3.15)

# Definiert den Projektnamen, die Version und die verwendete Programmiersprache (C++).
project(MeinProjekt VERSION 1.0 LANGUAGES CXX)

# Erstellt eine ausf�hrbare Datei.
# Der Name der App wird "App" sein, erstellt aus der Datei "main.cpp".
# Wenn du mehr Dateien hast (z.B. utils.cpp), f�ge sie einfach hinzu:
# add_executable(App main.cpp utils.cpp)
add_executable(App main.cpp)

# Der moderne Weg, um den C++-Standard festzulegen (hier C++20).
# Das ist besser als globale Variablen zu setzen.
target_compile_features(App PRIVATE cxx_std_20)
