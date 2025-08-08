#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <random>
#include <memory>
#include <algorithm>
#include <string>
#include <iomanip>
#include <limits>

// Definition von M_PI, falls nicht vorhanden
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Konfiguration
// Diese Dimensionen definieren die Weltgrenzen für die Physikberechnung
const unsigned int WORLD_WIDTH = 1200;
const unsigned int WORLD_HEIGHT = 800;
// Entropie-Auflösung
const unsigned int ENTROPY_GRID_W = 120;
const unsigned int ENTROPY_GRID_H = 80;
const float PHYSICS_TIMESTEP = 0.01f;
const int EVOLUTION_INTERVAL = 500; // Schritte zwischen Evolutionen
const int MAX_SIMULATION_STEPS = 5000; // Begrenzung der Simulationsdauer für die Textausgabe
const int OUTPUT_INTERVAL = 100; // Ausgabe alle X Schritte

// Setup für Zufallszahlen
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

// ============================================================================
// Hilfsstruktur für 2D Vektoren (Ersatz für sf::Vector2f)
// ============================================================================
struct Vector2f {
    float x, y;

    Vector2f() : x(0.0f), y(0.0f) {}
    Vector2f(float x, float y) : x(x), y(y) {}

    Vector2f operator+(const Vector2f& other) const { return { x + other.x, y + other.y }; }
    Vector2f operator-(const Vector2f& other) const { return { x - other.x, y - other.y }; }
    Vector2f operator*(float scalar) const { return { x * scalar, y * scalar }; }
    Vector2f operator/(float scalar) const { return { x / scalar, y / scalar }; }
    void operator+=(const Vector2f& other) { x += other.x; y += other.y; }
};

// Hilfsfunktionen für Vektoren
float length(const Vector2f& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}
float lengthSq(const Vector2f& v) {
    return v.x * v.x + v.y * v.y;
}
Vector2f normalize(const Vector2f& v) {
    float len = length(v);
    if (len > 1e-6f)
        return v / len;
    return { 0.0f, 0.0f };
}

// ============================================================================
// 1. Entropie-Generator (Wellenmuster)
// ============================================================================
class WaveEntropyGenerator {
private:
    float time = 0.0f;
    std::vector<float> entropyField;

public:
    WaveEntropyGenerator() {
        entropyField.resize(ENTROPY_GRID_W * ENTROPY_GRID_H);
    }

    void update(float dt) {
        time += dt;

        // Berechne das Entropiefeld in der Gitterauflösung
        for (unsigned int y = 0; y < ENTROPY_GRID_H; ++y) {
            for (unsigned int x = 0; x < ENTROPY_GRID_W; ++x) {
                // Normalisierte Koordinaten (0 bis 1)
                float nx = (float)x / ENTROPY_GRID_W;
                float ny = (float)y / ENTROPY_GRID_H;

                // Komplexe Welleninterferenz
                float wave1 = sin(nx * 15.0f + time * 0.8f);
                float wave2 = cos(ny * 12.0f + time * 0.5f);

                // Zirkulare Welle
                float dx = nx - 0.5f;
                float dy = ny - 0.5f;
                float dist = sqrt(dx * dx + dy * dy);
                float wave3 = sin(dist * 25.0f - time * 1.2f);

                // Normalisiere auf 0 bis 1
                float entropy = (wave1 + wave2 + wave3 + 3.0f) / 6.0f;
                entropyField[y * ENTROPY_GRID_W + x] = entropy;
            }
        }
    }

    // Holt den Entropiewert an einer Weltposition durch Sampling des Gitters
    float getEntropyAt(const Vector2f& position) const {
        // Berechne die Gitterkoordinaten
        int x = static_cast<int>((position.x / WORLD_WIDTH) * ENTROPY_GRID_W);
        int y = static_cast<int>((position.y / WORLD_HEIGHT) * ENTROPY_GRID_H);

        // Sicherstellen, dass die Indizes gültig sind (wichtig wegen Torus-Logik am Rand)
        x = std::clamp(x, 0, (int)ENTROPY_GRID_W - 1);
        y = std::clamp(y, 0, (int)ENTROPY_GRID_H - 1);

        return entropyField[y * ENTROPY_GRID_W + x];
    }

    // Funktion zur Ausgabe der durchschnittlichen Entropie für die Analyse
    float getAverageEntropy() const {
        float sum = 0.0f;
        for (float val : entropyField) {
            sum += val;
        }
        return sum / entropyField.size();
    }
};

// ============================================================================
// 2. Kausalnetz-Struktur
// ============================================================================
class CausalNode;
struct CausalLink {
    std::weak_ptr<CausalNode> source;
    std::weak_ptr<CausalNode> target;
    float weight; // Stärke des kausalen Einflusses
    int id;
};

class CausalNode : public std::enable_shared_from_this<CausalNode> {
public:
    int id;
    Vector2f position;
    Vector2f velocity;
    float mass;

    CausalNode(int id, Vector2f pos, float m = 1.0f) : id(id), position(pos), velocity({ 0.0f, 0.0f }), mass(m) {}

    // Berechnet Kräfte und aktualisiert die Geschwindigkeit (Semi-impliziter Euler Schritt 1)
    void calculateInfluenceAndVelocity(const std::vector<CausalLink>& links, const WaveEntropyGenerator& entropyGen) {
        Vector2f accumulatedForce = { 0.0f, 0.0f };

        for (const auto& link : links) {
            // Prüfe, ob dieser Knoten das Ziel des Links ist
            // Wir nutzen get() für den Vergleich der rohen Pointer
            if (link.target.lock().get() == this) {
                auto sourceNode = link.source.lock();
                if (sourceNode) {
                    // 1. Hole die lokale Entropie
                    float entropy = entropyGen.getEntropyAt(this->position);

                    // 2. Modifiziere den Einfluss: Hohe Entropie schwächt Kausalität.
                    float causality_factor = 1.0f - entropy;
                    // Füge eine Mindestkausalität hinzu, damit das System nicht stoppt
                    causality_factor = std::max(0.1f, causality_factor);

                    float effectiveWeight = link.weight * causality_factor;

                    // 3. Wende die physikalische Regel an (Gravitationsähnlich)
                    Vector2f direction = sourceNode->position - this->position;
                    float distanceSq = lengthSq(direction);

                    if (distanceSq > 1.0f) { // Vermeide Singularitäten
                        float distance = std::sqrt(distanceSq);
                        Vector2f normalizedDirection = direction / distance;

                        // Kraft = (G_effektiv * M1 * M2 / R^2)
                        float forceMagnitude = (effectiveWeight * this->mass * sourceNode->mass) / distanceSq;
                        accumulatedForce += normalizedDirection * forceMagnitude;
                    }
                }
            }
        }

        // Wende Kraft an (a=F/m) und integriere Geschwindigkeit
        // v(t+dt) = v(t) + a(t) * dt
        velocity += accumulatedForce * (PHYSICS_TIMESTEP / this->mass);
    }

    // Aktualisiert die Position (Semi-impliziter Euler Schritt 2)
    void updatePosition() {
        // Integriere Position mit der NEUEN Geschwindigkeit
        // p(t+dt) = p(t) + v(t+dt) * dt
        position += velocity * PHYSICS_TIMESTEP;

        // Begrenze die Position (Torus-Welt, Objekte erscheinen auf der anderen Seite)
        if (position.x < 0) position.x = WORLD_WIDTH;
        if (position.x > WORLD_WIDTH) position.x = 0;
        if (position.y < 0) position.y = WORLD_HEIGHT;
        if (position.y > WORLD_HEIGHT) position.y = 0;
    }
};

class EvolvingCausalNetwork {
public:
    std::vector<std::shared_ptr<CausalNode>> nodes;
    std::vector<CausalLink> links;
    int nextLinkId = 0;

    void addNode(std::shared_ptr<CausalNode> node) {
        nodes.push_back(node);
    }

    void addLink(std::shared_ptr<CausalNode> source, std::shared_ptr<CausalNode> target, float weight) {
        links.push_back({ source, target, weight, nextLinkId++ });
    }

    // Wendet die vom NN berechneten Updates an
    void applyEvolution(const std::map<int, float>& weightUpdates) {
        for (auto& link : links) {
            if (weightUpdates.count(link.id)) {
                link.weight = weightUpdates.at(link.id);
                // Begrenze die Gewichte, um Explosionen zu vermeiden
                link.weight = std::clamp(link.weight, 0.01f, 100.0f);
            }
        }
    }
};

// 3. Neuronales Netz (Evolver) - Platzhalter
class NeuralEvolver {
public:
    // simuliert die analyse und evolution durch ein NN.
    void processAndEvolve(EvolvingCausalNetwork& network, const WaveEntropyGenerator& entropyGen) {
        std::cout << "\n*** [EVOLUTION EVENT] ***\n";
        std::cout << "[NN] Analyzing Causal Structure and applying evolutionary pressure...\n";

        // PLATZHALTER-Logik fuer das neuronale netz
        std::map<int, float> proposedUpdates;

        float max_change = 0.0f;

        for (const auto& link : network.links) {
            auto targetNode = link.target.lock();
            if (!targetNode) continue;

            float entropy = entropyGen.getEntropyAt(targetNode->position);
            float currentWeight = link.weight;

            // regel 1: exploration (zufaellige mutation, verstaerkt durch hohe entropie)
            // staerkere mutationen in chaotischen (hohe entropie) regionen.
            float exploration = (dis(gen) - 0.5f) * 10.0f * entropy;

            // regel 2: exploitation (simulierte fitnessfunktion: stabilitätssuche)
            // reduziere gewicht, wenn geschwindigkeit des ziels hoch ist (instabilität).
            float velocityMag = length(targetNode->velocity);
            float stability_adjustment = 0.0f;
            if (velocityMag > 150.0f) {
                stability_adjustment = -5.0f; // Dämpfen
            }
            else if (velocityMag < 20.0f) {
                stability_adjustment = 2.0f; // Verstärken, wenn zu langsam
            }

            float newWeight = currentWeight + exploration + stability_adjustment;
            proposedUpdates[link.id] = newWeight;

            float change = std::abs(newWeight - currentWeight);
            if (change > max_change) {
                max_change = change;
            }
        }

        network.applyEvolution(proposedUpdates);
        std::cout << "[NN] Evolution applied. Max weight change observed: " << std::fixed << std::setprecision(3) << max_change << "\n";
        std::cout << "*************************\n\n";
    }
};

// engine und mainclass 
class PhysicsEngine {
private:
    EvolvingCausalNetwork causalNetwork;
    WaveEntropyGenerator entropyGenerator;
    NeuralEvolver neuralEvolver;
    long long stepCount = 0;

public:
    PhysicsEngine() {}

    void initializeSimulation() {
        // erstelle ein system aus mehreren koerpern
        // Zentralmasse
        auto center = std::make_shared<CausalNode>(0, Vector2f{ WORLD_WIDTH / 2.0f, WORLD_HEIGHT / 2.0f }, 10000.0f);
        causalNetwork.addNode(center);

        // orbitierende objekte lol
        float G = 50.0f; // Initiale Gravitationskonstante (Kausalgewicht)
        int num_planets = 6;

        for (int i = 1; i <= num_planets; ++i) {
            float angle = (2 * M_PI * i) / num_planets;
            float distance = 100.0f + i * 50.0f;
            Vector2f pos = { center->position.x + distance * cos(angle),
                            center->position.y + distance * sin(angle) };
            auto planet = std::make_shared<CausalNode>(i, pos, 50.0f);

            // berechnet theoretische orbitalgeschwindigkeit (v = sqrt(GM/r)) für den start
            float speed = sqrt((G * center->mass) / distance);
            planet->velocity = Vector2f{ -speed * sin(angle), speed * cos(angle) };

            causalNetwork.addNode(planet);
            // Kausale Links (Gravitation ist bidirektional)
            causalNetwork.addLink(center, planet, G);
            causalNetwork.addLink(planet, center, G);
        }
        std::cout << "Simulation initialized with " << causalNetwork.nodes.size() << " nodes and " << causalNetwork.links.size() << " links.\n";
    }

    void run() {
        //  mainloop fuer feste anzahl an steps
        while (stepCount < MAX_SIMULATION_STEPS) {
            // handleEvents() entfernt
            update();

            if (stepCount % OUTPUT_INTERVAL == 0) {
                printState();
            }
        }
        std::cout << "\nSimulation finished after " << MAX_SIMULATION_STEPS << " steps.\n";
    }

private:
    void update() {
        stepCount++;

        // 1.  aktualisieren des entropiefeldes
        entropyGenerator.update(PHYSICS_TIMESTEP);

        // 2.  physics updates
        // Schritt 2a: Berechne Kräfte und aktualisiere Geschwindigkeit
        for (auto& node : causalNetwork.nodes) {
            node->calculateInfluenceAndVelocity(causalNetwork.links, entropyGenerator);
        }

        // Schritt 2b: Aktualisiere Position basierend auf neuer Geschwindigkeit
        for (auto& node : causalNetwork.nodes) {
            node->updatePosition();
        }

        // 3.  evolution des netzes
        if (stepCount > 0 && stepCount % EVOLUTION_INTERVAL == 0) {
            neuralEvolver.processAndEvolve(causalNetwork, entropyGenerator);
        }
    }

    //  renderfunktion durch text fuer only text ersetzt
    void printState() {
        std::cout << "\n=== Simulation Step: " << stepCount << " ===\n";
        std::cout << "Global Average Entropy: " << std::fixed << std::setprecision(4) << entropyGenerator.getAverageEntropy() << "\n";

        // output der node position, geschwindigkeit und local entropy
        std::cout << "--- Nodes (ID: PosX, PosY | Vel Magnitude | Local Entropy) ---\n";
        for (const auto& node : causalNetwork.nodes) {
            float velocityMag = length(node->velocity);
            float localEntropy = entropyGenerator.getEntropyAt(node->position);
            // Formatierte Ausgabe für bessere Lesbarkeit in Spalten
            std::cout << "Node " << std::setw(2) << node->id << ": "
                << std::fixed << std::setprecision(2) << std::setw(7) << node->position.x << ", " << std::setw(7) << node->position.y
                << " | Vel: " << std::setw(8) << velocityMag
                << " | Entropy: " << std::fixed << std::setprecision(4) << localEntropy << "\n";
        }

        // zusammenfassen der link weights fuer statistik
        float min_weight = std::numeric_limits<float>::max();
        float max_weight = std::numeric_limits<float>::min();
        float avg_weight = 0.0f;

        if (!causalNetwork.links.empty()) {
            for (const auto& link : causalNetwork.links) {
                avg_weight += link.weight;
                if (link.weight < min_weight) min_weight = link.weight;
                if (link.weight > max_weight) max_weight = link.weight;
            }
            avg_weight /= causalNetwork.links.size();
        }
        else {
            min_weight = 0.0f;
            max_weight = 0.0f;
        }

        std::cout << "--- Links Summary ---\n";
        std::cout << "Count: " << causalNetwork.links.size() << "\n";
        std::cout << "Weight Range: [" << std::fixed << std::setprecision(3) << min_weight << " - " << max_weight << "]\n";
        std::cout << "Average Weight: " << avg_weight << "\n";
        std::cout << "==========================\n";
    }
};

// mainloop fuer die engine  
int main() {
    try {
        PhysicsEngine engine;
        engine.initializeSimulation();
        std::cout << "Starting Evolving Causal Physics Simulation (Text Mode)...\n";
        engine.run();
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}