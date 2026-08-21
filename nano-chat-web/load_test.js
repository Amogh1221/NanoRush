const url = "https://amogh1221-nanorush.hf.space/chat";
const payload = {
    messages: [{ role: "user", content: "Write a very short poem about the ocean." }]
};

const CONCURRENT_USERS = 5;

async function simulateUser(userId) {
    const start = Date.now();
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`Status ${res.status}`);

        // We only care about the time to get the full response for load testing
        const text = await res.text();
        const duration = (Date.now() - start) / 1000;
        
        console.log(`✅ User ${userId} finished in ${duration.toFixed(2)}s`);
        return true;
    } catch (e) {
        console.log(`❌ User ${userId} FAILED: ${e.message}`);
        return false;
    }
}

async function runLoadTest() {
    console.log(`🚀 Starting load test with ${CONCURRENT_USERS} concurrent users...`);
    const startTotal = Date.now();
    
    // Create an array of promises for concurrent execution
    const tasks = [];
    for (let i = 1; i <= CONCURRENT_USERS; i++) {
        tasks.push(simulateUser(i));
    }
    
    // Wait for all users to finish
    const results = await Promise.all(tasks);
    
    const successes = results.filter(r => r).length;
    const totalTime = (Date.now() - startTotal) / 1000;
    
    console.log(`\n📊 RESULTS:`);
    console.log(`Total Time: ${totalTime.toFixed(2)}s`);
    console.log(`Successful Requests: ${successes} / ${CONCURRENT_USERS}`);
}

runLoadTest();
