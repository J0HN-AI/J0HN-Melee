"use client";

import { useEffect, useState } from "react";
import { io } from "socket.io-client";

//const socket = io("http://localhost:9500");

function convert_2_time(game_time) {
    const date = new Date(game_time * 1000);
    
    let minutes = date.getMinutes();
    let seconds = date.getSeconds();
    let millis = date.getMilliseconds();

    let time_str = minutes.toString() + ":" + 
               seconds.toString() + ":" +
               millis.toString()

    return time_str    
}

export default function Home() {
    const [maxLogs, setMaxLogs] = useState(5);
    const [logs, setLogs] = useState({
        "5000": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5001": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5002": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5003": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5004": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5005": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5006": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5007": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5008": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5009": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
        "5010": {
            "game": 14,
            "game_time": 12.435,
            "stage": "FINAL_DESTINATION",
            "agent": "KIRBY",
            "agent_percent": 14,
            "agent_stock": 4,
            "CPU": "STEVE",
            "CPU_percent": 145,
            "CPU_stock": 4,
            "score": -2.43456,
            "avg_score": 3.23456,
            "learn_iters": 6547
        },
    });

    const sorted_instances = Object.entries(logs).sort().slice(0, maxLogs);

    return (
        <div className="font-sans min-h-screen bg-gray-950">
            <div className="bg-gray-800 shadow p-4 flex justify-between items-center">
                <h1 className="text-xl font-bold">J0HNMELEE Instances Viewer</h1>
                <div className="flex items-center space-x-2">
                    <label htmlFor="max-logs" className="text-sm font-medium">
                        Max Number of logs:
                    </label>
                    <input
                        id="max-logs"
                        type="number"
                        min="1"
                        max={Object.keys(logs).length}
                        value={maxLogs}
                        onChange={(e) => setMaxLogs(Number(e.target.value))}
                        className="border border-indigo-600 rounded px-2 py-1 w-20 text-sm"
                    />
            </div>
        </div>
            <div className="p-4 grid grid-cols-1 gap-4">
                {sorted_instances.map(([id, game], index) => (
                    <div key={id} className="border rounded-2xl shadow p-4 bg-gray-800 border-indigo-800">
                        <h2 className="text-xl font-bold mb-2">Game ID: {id}</h2>
                        <div className="grid grid-cols-12 gap-2 text-sm">
                            {Object.entries(game).map(([key, value]) => (
                            <div key={key} className="flex justify-center flex-col">
                                <span className="font-medium capitalize mb-2">{key.replaceAll('_', ' ')}:</span>
                                {key == "game_time" ? <span>{convert_2_time(value)}</span> : <span>{value}</span>}
                            </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}