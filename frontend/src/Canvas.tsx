// external
import React, { useRef, useEffect, useState } from 'react';

interface Coordinate {
    row: number,
    col: number,
}

interface CanvasProps {
    handlePredict: (pixels: number[][]) => void;
    gridSize: number;
    pixelSize: number;
    onClear?: () => void;           
}


export default function Canvas({ gridSize, pixelSize, handlePredict, onClear}: CanvasProps) {
    const [lastPos, setLastPos] = useState<Coordinate | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const GRID_SIZE = gridSize;
    const PIXEL_SIZE = pixelSize;
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [pixels, setPixels] = useState<number[][]>(
        Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0))
    );

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        for (let row = 0; row < GRID_SIZE; row++) {
            for (let col = 0; col < GRID_SIZE; col++) {
                ctx.fillStyle = pixels[row][col] === 0 ? '#FFFFFF' : '#000000';
                ctx.fillRect(col * PIXEL_SIZE, row * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
                ctx.strokeStyle = '#DDD';
                ctx.strokeRect(col * PIXEL_SIZE, row * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
            }
        }
    }, [pixels, GRID_SIZE, PIXEL_SIZE]);
    
    function drawPixelAt(row: number, col: number) {
        setPixels(prev => {
            const next = prev.map(arr => [...arr]);
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const r = row + dr;
                    const c = col + dc;
                    if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
                        if (dr === 0 && dc === 0) {
                            next[r][c] = 1;
                        } else if (Math.abs(dr) + Math.abs(dc) === 1) {
                            next[r][c] = Math.max(next[r][c], 0.45);
                        } else {
                            next[r][c] = Math.max(next[r][c], 0.2);}}}}
        return next;});}


    // Bresenham's line algorithm for pixel interpolation
    function drawLine(from: Coordinate, to: Coordinate) {
        let x0 = from.col, y0 = from.row, x1 = to.col, y1 = to.row;
        const dx = Math.abs(x1 - x0), dy = Math.abs(y1 - y0);
        const sx = x0 < x1 ? 1 : -1;
        const sy = y0 < y1 ? 1 : -1;
        let err = dx - dy;
        while (true) {
            drawPixelAt(y0, x0);
            if (x0 === x1 && y0 === y1) break;
            const e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    function getRowCol(e: React.MouseEvent<HTMLCanvasElement>): Coordinate {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const col = Math.floor(x / PIXEL_SIZE);
        const row = Math.floor(y / PIXEL_SIZE);
        return { row, col };
    }

    function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
        setIsDrawing(true);
        const pos = getRowCol(e);
        drawPixelAt(pos.row, pos.col);
        setLastPos(pos);
    }

    function handleMouseUp() {
        setIsDrawing(false);
        setLastPos(null);
    }

    function handleMouseLeave() {
        setIsDrawing(false);
        setLastPos(null);
    }

    function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
        if (isDrawing) {
            const pos = getRowCol(e);
            if (lastPos) {
                drawLine(lastPos, pos);
            } else {
                drawPixelAt(pos.row, pos.col);
            }
            setLastPos(pos);
        }
    }

    function clearPixels() {
        setPixels(s => s.map((arr) => arr.map(_ => 0)));
        if (onClear) {
            onClear();
        }
    }
    
    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <canvas
                ref={canvasRef}
                width={GRID_SIZE * PIXEL_SIZE}
                height={GRID_SIZE * PIXEL_SIZE}
                style={{ border: '1px solid #888', cursor: 'pointer' }}
                onMouseDown={handleMouseDown}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
                onMouseMove={handleMouseMove}
            />
            <div>
                <button type="button" onClick={() => { handlePredict(pixels) }}>Predict</button>
                <button type="button" onClick={() => { clearPixels() }}>Clear</button>
            </div>
        </div>
    );
}
