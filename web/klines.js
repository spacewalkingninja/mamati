Storage.prototype.setObject = function(key, value) {
    this.setItem(key, JSON.stringify(value));
}
 
Storage.prototype.getObject = function(key) {
    var value = this.getItem(key);
    return JSON.parse(value);
}



const width  = window.innerWidth || document.documentElement.clientWidth || 
document.body.clientWidth;
var canvas = document.getElementById("chart");  
var canvas2 = document.getElementById("chart2");  
canvas.width = width*.8;
canvas2.width = width*.8;

let cursorX = null;

const verticalLinePlugin = {
    id: 'verticalLinePlugin',
    afterDraw(chart, args, options) {
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;

        if (cursorX !== null) {
            // Save the current canvas state
            ctx.save();

            // Draw the vertical line
            ctx.beginPath();
            ctx.moveTo(cursorX, chartArea.top);
            ctx.lineTo(cursorX, chartArea.bottom);
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Restore the canvas state
            ctx.restore();
        }
    }
};

const slopeAngles = {
    id: 'slopeAnglePlugin',
    afterDraw: (chart) => {

        const ctx = chart.ctx;
        const d_macd_data = chart.data.datasets.find(dataset => dataset.label === 'DIF').data;
        const latestIndex = d_macd_data.length - 1;

        // Find the latest inversion
        let latestInversionIndex = latestIndex;
        for (let i = latestIndex; i >= 1; i--) {
            if ((d_macd_data[i].y > 0 && d_macd_data[i - 1].y <= 0) || (d_macd_data[i].y < 0 && d_macd_data[i - 1].y >= 0)) {
                latestInversionIndex = i;
                break;
            }
        }

        // Find the absolute peak (maximum or minimum) since the latest inversion
        let latestPeakIndex = latestInversionIndex;
        for (let i = latestInversionIndex; i <= latestIndex; i++) {
            if (Math.abs(d_macd_data[i].y) > Math.abs(d_macd_data[latestPeakIndex].y)) {
                latestPeakIndex = i;
            }
        }

        if (latestPeakIndex === latestInversionIndex) {
            console.warn('No peak found since the latest inversion.');
            return;
        }

        const latestPeak = d_macd_data[latestPeakIndex];
        const latestPoint = d_macd_data[latestIndex];

        // Calculate the slope angle using pixel values
        const peakX = chart.scales.x.getPixelForValue(latestPeak.x);
        const peakY = chart.scales.y.getPixelForValue(latestPeak.y);
        const pointX = chart.scales.x.getPixelForValue(latestPoint.x);
        const pointY = chart.scales.y.getPixelForValue(latestPoint.y);

        const deltaX = pointX - peakX;
        const deltaY = pointY - peakY;
        const angle = Math.atan2(deltaY, deltaX);

        // Draw the extrapolated line
        const lineLength = 100; // Length of the extrapolated line
        const lineEndX = pointX + lineLength * Math.cos(angle);
        const lineEndY = pointY + lineLength * Math.sin(angle);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(pointX, pointY);
        ctx.lineTo(lineEndX, lineEndY);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Calculate the radius for the circles to intersect correctly
        const radius = Math.hypot(deltaX, deltaY) / 2;

        // Calculate the centers of the circles
        const midX = (peakX + pointX) / 2;
        const midY = (peakY + pointY) / 2;
        const offset = Math.sqrt(radius * radius - Math.pow(midX - peakX, 2));

        const circle1CenterY = midY - offset;
        const circle2CenterY = midY + offset;

        // Draw the circles
        ctx.beginPath();
        ctx.arc(midX, circle1CenterY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(midX, circle2CenterY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 1;
        ctx.stroke();


        ctx.restore();
    }
}

const fibplugin = {
    id: 'fibonacciTimeZones',
    afterDatasetsDraw: function(chart) {
        const ctx = chart.ctx;
        const dataset = chart.data.datasets.find(ds => ds.label === 'Raw Signal');
        if (!dataset) return;
        
        console.warn("SWINGS")
        let tlfs = 0 
        for (let tlf = 0.01; tlf < 0.2 ; tlf+=0.01) {
            tlfs++
            const swings = identifySwingsWithDynamicTolerance(dataset.data, 5, tlf);
            //console.warn(swings)
            if (swings.length < 2) return;

            // Find the largest downswing
            let maxDownswing = 0;
            let initialSwing = null;
            let secondSwing = null;
    
            for (let i = 0; i < swings.length - 1; i++) {
                if (swings[i].y > swings[i + 1].y) {
                    let downswing = swings[i].y - swings[i + 1].y;
                    if (downswing > maxDownswing) {
                        maxDownswing = downswing;
                        initialSwing = swings[i];
                        secondSwing = swings[i + 1];
                    }
                }
            }
    
            if (!initialSwing || !secondSwing) return;
    
            const fibSequence = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
    
            ctx.save();
            ctx.lineWidth = 1;
            //ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba('+tlf*1000+', 0, '+ parseInt((255/tlfs)*1.8)+', 0.75)';
    
            fibSequence.forEach(num => {
                const timeDelta = secondSwing.x - initialSwing.x;
                const futureTime = secondSwing.x + num * timeDelta;
                const futurePoint = chart.scales.x.getPixelForValue(futureTime);
    
                ctx.beginPath();
                ctx.moveTo(futurePoint, chart.chartArea.top);
                ctx.lineTo(futurePoint, chart.chartArea.bottom);
                ctx.stroke();
            });
        }

        ctx.restore();
    }
};

// Function to identify swings
function identifySwingsWithDynamicTolerance(data, lookback = 5, toleranceFactor = 0.1) {
    const swings = [];
    const dataRange = Math.max(...data.map(point => point.y)) - Math.min(...data.map(point => point.y));
    const tolerance = toleranceFactor * dataRange;

    for (let i = lookback; i < data.length - lookback; i++) {
        let isSwingHigh = true;
        let isSwingLow = true;

        for (let j = 1; j <= lookback; j++) {
            if (data[i].y <= data[i - j].y + tolerance && data[i].y <= data[i + j].y + tolerance) {
                isSwingHigh = false;
            }
            if (data[i].y >= data[i - j].y - tolerance && data[i].y >= data[i + j].y - tolerance) {
                isSwingLow = false;
            }
        }

        if (isSwingHigh || isSwingLow) {
            swings.push(data[i]);
        }
    }
    return swings;
}
/*
var imgplugin = {
    id: 'customCanvasBackgroundImage',
    beforeDraw: (chart) => {
      const ctx = chart.ctx;
      const { top, left, bottom, right, width, height } = chart.chartArea;
  
      if (image.complete) {
        ctx.save(); // Save the current state
        ctx.globalAlpha = 0.5; // Set the transparency if desired
        ctx.drawImage(image, left, top, right - left, bottom - top); // Draw the image stretched to chart area
        ctx.restore(); // Restore the original state
      } else {
        image.onload = () => chart.draw(); // Redraw the chart once the image is loaded
      }
    }
  };
*/

let image = new Image();
function genBlinds(chart, heatmapData, data) {
    const ctx = chart.ctx;
                const { top, left, bottom, right, width, height } = chart.chartArea;
                image = null;
                image = new Image();
                console.log("GENIMG")
                
                const heatData = heatmapData.liqHeatMap.data;
                const chartTimeArray = heatmapData.liqHeatMap.chartTimeArray;
                const priceArray = heatmapData.liqHeatMap.priceArray;

                // Find min and max values
                let minValue = Number.POSITIVE_INFINITY;
                let maxValue = Number.NEGATIVE_INFINITY;

                let newTimeArray = []
                let stepsPassed = 0
                let l = Object.keys(data['KLINES']).length
                let firstData = data['KLINES'][0].x
                //let lastData = data['KLINES'][l].x
                let maxData = 0
                console.log(chart.scales.y.min)
                console.log(chart.scales.y.max)
                let ni = 0;
                for (let i = 0; i < chartTimeArray.length; i++) {
                    if(chartTimeArray[i]>=firstData)
                        {
                            newTimeArray[ni] = chartTimeArray[i]
                            ni++;
                            if(stepsPassed == 0)
                                {
                                    stepsPassed = i-1
                                }
                            if(i==chartTimeArray.length-1) {
                                maxData = chartTimeArray[i]
                            }
                        }
                }

                let nl = 0;
                for (let i = 0; i < l; i++) {
                    if(data['KLINES'][i].x > maxData){
                        nl = i;
                        break;
                    }
                }
                let nSTEP = width / l
                let nWidth = nSTEP * nl
                let newPriceArray = []
                let pi = 0;
                let newPriceMin = chart.scales.y.min
                let newPriceMax = chart.scales.y.max
                let priceMinStep = 0
                let priceMaxStep = 0
                for (let i = 0; i < priceArray.length; i++) {
                    if(priceArray[i]>=newPriceMin)
                        {
                            newPriceArray[pi] = priceArray[i]
                            pi++;
                            if(priceMinStep == 0)
                                {
                                    priceMinStep = i
                                }
                            if(priceArray[i]>=newPriceMax){
                                priceMaxStep = i
                                break;
                            }
                        }
                }

                for (let i = 0; i < heatData.length; i++) {
                        const value = parseFloat(heatData[i][2]);
                        if (value < minValue) minValue = value;
                        if (value > maxValue) maxValue = value;
                }
                //const width = 1920
                //const height = 420
                const timeStep = nWidth / (newTimeArray.length - 1);
                const timeStep2 = width / (newTimeArray.length - 1);
                const priceStep = height / (newPriceArray.length - 1);

                // Create an offscreen canvas
                const offscreenCanvas = document.createElement('canvas');
                offscreenCanvas.width = width;
                offscreenCanvas.height = height;
                const offscreenCtx = offscreenCanvas.getContext('2d');
                //attentionArray=[]
                //aAi=0
                // Draw the heatmap on the offscreen canvas
                for (let i = 0; i < heatData.length; i++) {
                    const value = parseFloat(heatData[i][2]);
/* REMOVE? This prolly wont be used this way because performance reasons 
                    tx=((heatData[i][0]-stepsPassed-1) * timeStep)+timeStep2
                    if(heatData[i][0]>=stepsPassed && tx>nWidth)
                        {
                            const normalizedValue = (value - minValue) / (maxValue - minValue);
                            attentionArray[aAi]=[normalizedValue, priceArray[heatData[i][1]]]
                            aAi++
                        }
*/
                    if(heatData[i][0]>=stepsPassed &&
                       heatData[i][1]>=priceMinStep &&
                       heatData[i][1]<=priceMaxStep 
                        )
                        {
                            let color = getHeatmapColor(value, minValue, maxValue, 1.2);
                            offscreenCtx.fillStyle = color;
                            offscreenCtx.fillRect((heatData[i][0]-stepsPassed-1) * timeStep2, height - (heatData[i][1]-priceMinStep) * priceStep, timeStep2, priceStep);

                            color = getHeatmapColor(value, minValue, maxValue);
                            offscreenCtx.fillStyle = color;
                            offscreenCtx.fillRect((heatData[i][0]-stepsPassed-1) * timeStep, height - (heatData[i][1]-priceMinStep) * priceStep, timeStep, priceStep);
                        }
                }
                // Create an image from the offscreen canvas
                image.src = offscreenCanvas.toDataURL();
                if (image.complete) {
                    ctx.save(); // Save the current state
                    ctx.globalAlpha = .8; // Set the transparency if desired
                    ctx.drawImage(image, left, top, right - left, bottom - top); // Draw the image stretched to chart area
                    
                    ctx.restore(); // Restore the original state
                } else {
                    //console.warn(attentionArray)
                    image.onload = () => chart.draw(); // Redraw the chart once the image is loaded
                }
}

function getHeatmapColor(value, minValue, maxValue, s=1) {
    const normalizedValue = (value - minValue) / (maxValue - minValue);

    const g = normalizedValue < 0.5 ? Math.floor(255 * (2*s) * normalizedValue) : 255;
    const b = normalizedValue < 0.5 ? 0 : Math.floor(255 * ((2*s) * (normalizedValue - 0.5)));
    const r = normalizedValue < 0.5 ? 0 : Math.floor(255 * ((2*s) * (normalizedValue - 0.25)));
    const a = 1;

    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }

heatmapData = false


        
function calculateFibonacciLevels(ahigh, alow) {
    let high = Math.max(...ahigh);
    let low = Math.min(...alow);
    //let diff = maxPrice - minPrice;
    //console.log(diff)
    let diff = high - low;

    return {
        "0%": high,
        "23.6%": high - (diff * 0.236),
        "38.2%": high - (diff * 0.382),
        "50%": high - (diff * 0.5),
        "61.8%": high - (diff * 0.618),
        "100%": low
    };
}


function randomBar(target, index, date, lastClose) {
    var open = +klines[index].o;
    var close = +klines[index].c;
    var high = +klines[index].h;
    var low = +klines[index].l;

    if (!target[index]) {
        target[index] = {};
    }

    Object.assign(target[index], {
        x: klines[index].x,
        o: open,
        h: high,
        l: low,
        c: close
    });

    }
function getRandomData(dateStr, fibonacciLevels, lastTimestamp, barData, lineData, d_sma, d_ema, d_rsi, d_macd, d_signal, d_raw_signal, d_histogram, d_bb_sma, d_upper_bb, d_lower_bb, d_roc, d_stochastic_k, d_stochastic_d, d_vol, d_vpt, d_obv, d_kci, d_dpv, d_apv, d_anv) {
    date = luxon.DateTime.fromSeconds(lastTimestamp);
    //item.x = date.toLocaleDateString(); 
    console.log("GRD")
    for (let i = 0; i < l;) {
        date = date.plus({minutes: 1});
        //if (date.weekday <= 5) {
            randomBar(barData, i, date, i === 0 ? 30 : barData[i - 1].c);
            lineData[i] = {x: barData[i].x, y: barData[i].c};
            d_sma[i] = {x: barData[i].x, y: sma[i]> 0 ? sma[i] : null};
            d_ema[i] = {x: barData[i].x, y: ema[i]};
            d_rsi[i] = {x: barData[i].x, y: rsi[i]};
            d_macd[i] = {x: barData[i].x, y: macd[i]};
            d_signal[i] = {x: barData[i].x, y: signal[i]};
            d_raw_signal[i] = {x: barData[i].x, y: raw_signal[i]};
            d_histogram[i] = {x: barData[i].x, y: histogram[i]};
            d_bb_sma[i] = {x: barData[i].x, y: bb_sma[i]> 0 ? bb_sma[i] : null};
            d_upper_bb[i] = {x: barData[i].x, y: upper_bb[i]> 0 ? upper_bb[i] : null};
            d_lower_bb[i] = {x: barData[i].x, y: lower_bb[i]> 0 ? lower_bb[i] : null};
            d_roc[i] = {x: barData[i].x, y: roc[i]};
            d_stochastic_k[i] = {x: barData[i].x, y: stochastic_k[i]};
            d_stochastic_d[i] = {x: barData[i].x, y: stochastic_d[i]};
            d_vol[i] = {x: barData[i].x, y: vol[i]};
            d_vpt[i] = {x: barData[i].x, y: vpt[i]};
            d_obv[i] = {x: barData[i].x, y: obv[i]};
            d_kci[i] = {x: barData[i].x, y: kci[i]};
            d_dpv[i] = {x: barData[i].x, y: dpv[i]};
            d_apv[i] = {x: barData[i].x, y: apv[i]};
            d_anv[i] = {x: barData[i].x, y: anv[i]};
            let high = []
            let low = []
            ammin = [i+1, 15]
            let mmin = Math.min(...ammin)
                for (z=i; z>i-mmin;z--) {
                    high.push(+klines[i].h)
                    low.push(+klines[i].l)
                }
            let levels = calculateFibonacciLevels(high, low);
            fibonacciLevels.push({ x: klines[i].x, levels: levels });
            i++;
            
        //}
    }
}


function generateChart(data, heatmapData = false) {
    //const ctx2 = document.getElementById('myChart').getContext('2d');
    //getData = analyze_asset(0)

    

    if(!heatmapData){
        eel.get_heatmap()(function(heatmapData){
            heatmapData = JSON.parse(heatmapData)
            if(heatmapData){
                heatmapData = heatmapData.data;
            }
        })
    }

    
const imgplugin = {
    id: 'customCanvasBackgroundImage',
    beforeDraw: (chart) => {
        
    const ctx = chart.ctx;
    const { top, left, bottom, right, width, height } = chart.chartArea;
    console.log('imgpl')
    console.log(chart)

        heatmapData = window.localStorage.getObject('heatmapData')
        if(heatmapData){
            //heatmapData = heatmapData.data;
        

    
            console.log('IE')
            console.log(heatmapData)
                genBlinds(chart, heatmapData, data)
            
            
        
        }

        
        
    
    }
    
}
    
    
    //console.log(data)
    sma = data['SMA']
    ema = data['EMA']
    rsi = data['RSI']
    macd = data['MACD']
    signal = data['SIGNAL']
    raw_signal = data['RAW_S']
    raw_closing = data['RAW_CT']
    histogram = data['HISTOGRAM']
    bb_sma = data['BB_SMA']
    upper_bb = data['UPPER_BB']
    lower_bb = data['LOWER_BB']
    roc = data['ROC']
    stochastic_k = data['STOCHASTIC_K']
    stochastic_d = data['STOCHASTIC_D']
    klines = data['KLINES']
    vol = data['RAW_VOLUME']
    vpt = data['VPT']
    obv = data['OBV']
    kci = data['KCI']
    dpv = data['DPV']
    apv = data['PV']
    anv = data['NV']
    
    pred_ROC = data['pred_ROC']
    pred_SIGNAL = data['pred_SIGNAL']
    //pred_RAW_S = data['pred_RAW_S']
    pred_SMA = data['pred_SMA']
    pred_UPPER_BB = data['pred_UPPER_BB']
    pred_LOWER_BB = data['pred_LOWER_BB']
    pred_OBV = data['pred_OBV']
    pred_DPV = data['pred_DPV']

    predArr = [pred_ROC, pred_SIGNAL, pred_SMA, /*pred_RAW_S,*/ pred_UPPER_BB, pred_LOWER_BB, pred_OBV, pred_DPV]

    l = Object.keys(klines).length

    
    let fibonacciLevels = [];

    barCount = new Array(l)

    const lastItem = klines[0];
    const lastTimestamp = lastItem.x;

    //const interval = liTimestamp - lastTimestamp
    //const beginAt = bTimestamp

    
    //console.log(lastTimestamp)
    //console.log(vpt)
    var initialDateStr = new Date(lastTimestamp).toLocaleDateString(); 
    console.log(initialDateStr)

    function drawPreds(plevel, chart, yscale=chart.scales.y1) {
        let ai=1
        const dataset = chart.data.datasets[0].data
        const beginAt = chart.scales.x.getPixelForValue(dataset[dataset.length-1].x)
        const interval = beginAt - chart.scales.x.getPixelForValue(dataset[dataset.length-2].x)
        Object.keys(plevel).forEach(level => {
            ai++
            nlevel = plevel[level+1] ? level + 1 : level

            const startTime = beginAt+interval*ai;
            const futureTime = beginAt+interval*(ai+1);
            const startXPoint = startTime;
            const startYPoint = yscale.getPixelForValue(plevel[level]);
            const futureXPoint = futureTime;
            const futureYPoint = yscale.getPixelForValue(plevel[nlevel]);

            ctx.beginPath();
            ctx.moveTo(startXPoint, startYPoint);
            ctx.lineTo(futureXPoint, futureYPoint);
            ctx.stroke();
        });
    }

    function drawPreds(plevel, chart, yscale=chart.scales.y1, ctx) {
        let ai=1
        const dataset = chart.data.datasets[0].data
        const beginAt = chart.scales.x.getPixelForValue(dataset[dataset.length-1].x)
        const interval = beginAt - chart.scales.x.getPixelForValue(dataset[dataset.length-2].x)
        Object.keys(plevel).forEach(level => {
            ai++
            nlevel = plevel[level+1] ? level + 1 : level

            const startTime = beginAt+interval*ai;
            const futureTime = beginAt+interval*(ai+1);
            const startXPoint = startTime;
            const startYPoint = yscale.getPixelForValue(plevel[level]);
            const futureXPoint = futureTime;
            const futureYPoint = yscale.getPixelForValue(plevel[nlevel]);

            ctx.beginPath();
            ctx.moveTo(startXPoint, startYPoint);
            ctx.lineTo(futureXPoint, futureYPoint);
            ctx.stroke();
        });
    }

    function drawPredsInverted(plevel, chart4, yscale=chart4.scales.y1, ctx) {
        let ai=1
        const dataset = chart4.data.datasets[0].data
        const beginAt = chart4.scales.x.getPixelForValue(dataset[dataset.length-1].x)
        const interval = beginAt - chart4.scales.x.getPixelForValue(dataset[dataset.length-2].x)  
        Object.keys(plevel).forEach(level => {
            ai++
            nlevel = plevel[level+1] ? level + 1 : level

            const startTime = beginAt+interval*ai;
            const futureTime = beginAt+interval*(ai+1);
            const startXPoint = startTime;
            const startYPoint = yscale.getPixelForValue(plevel[level]);
            const futureXPoint = futureTime;
            const futureYPoint = yscale.getPixelForValue(plevel[nlevel]);

            ctx.beginPath();
            ctx.moveTo(startXPoint, startYPoint);
            ctx.lineTo(futureXPoint, futureYPoint);
            ctx.stroke();
        });
    }
    
    const predictionPluginChart = {
        id: 'predictionPluginChart',
        afterDatasetsDraw: function(chart) {
            const ctx = chart.ctx;
            //const dataset = chart.data.datasets.find(ds => ds.label === 'Raw Signal');
            //if (!dataset) return;
            
            ctx.save();
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(255, 191, 0, 0.75)';
            
            drawPreds(pred_OBV, chart, chart.scales.y2, ctx)

            ctx.strokeStyle = 'rgba(255, 0, 180, 0.75)';
            drawPreds(pred_DPV, chart, chart.scales.y3, ctx)
/*
            ctx.strokeStyle = 'rgba(255, 0, 10, 0.75)';
            drawPreds(pred_RAW_S, chart, chart.scales.y, ctx)
*/
            ctx.strokeStyle = 'rgba(150, 150, 110, 1)';
            drawPreds(pred_SMA, chart, chart.scales.y, ctx)

            ctx.strokeStyle = 'rgba(0, 150, 110, 0.75)';
            drawPreds(pred_LOWER_BB, chart, chart.scales.y, ctx)
            drawPreds(pred_UPPER_BB, chart, chart.scales.y, ctx)
            
            ctx.restore();
        }
    };

    const predictionPluginHelper = {
        id: 'predictionPluginHelper',
        afterDatasetsDraw: function(helperChart) {
            const ctx = helperChart.ctx;
            //const dataset = chart.data.datasets.find(ds => ds.label === 'Raw Signal');
            //if (!dataset) return;
            
            ctx.save();
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(0, 250, 10, 0.75)';
            console.log(helperChart)
            drawPreds(pred_ROC, helperChart, helperChart.scales.y3, ctx)
/*            ctx.strokeStyle = 'rgba(255, 0, 10, 0.75)';
            drawPreds(pred_RAW_S, helperChart, helperChart.scales.y, ctx)*/
            
            ctx.restore();
        }
    };

    const predictionPlugin4 = {
        id: 'predictionPlugin4',
        afterDatasetsDraw: function(chart4) {
            const ctx = chart4.ctx;
            //const dataset = chart.data.datasets.find(ds => ds.label === 'Raw Signal');
            //if (!dataset) return;
            //console.log(chart4)
            ctx.save();
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = 'rgba(255, 0, 180, 0.75)';
            drawPredsInverted(pred_DPV, chart4, chart4.scales.y, ctx)
            ctx.strokeStyle = 'rgba(255, 165, 0, 0.75)';
            drawPredsInverted(pred_SIGNAL, chart4, chart4.scales.y, ctx)
            
            ctx.restore();
        }
    };
    
    var ctx = document.getElementById('chart').getContext('2d');
    ctx.canvas.width = 1000;
    ctx.canvas.height = 250;

    var ctx2 = document.getElementById('chart2').getContext('2d');
    ctx2.canvas.width = 1000;
    ctx2.canvas.height = 250;

    var ctx4 = document.getElementById('chart4').getContext('2d');
    ctx4.canvas.width = 1000;
    ctx4.canvas.height = 250;

    var barData = new Array(barCount);
    var lineData = new Array(barCount);
    var d_sma = new Array(barCount);
    var d_ema = new Array(barCount);
    var d_rsi = new Array(barCount);
    var d_macd = new Array(barCount);
    var d_signal = new Array(barCount);
    var d_raw_signal = new Array(barCount);
    var d_histogram = new Array(barCount);
    var d_bb_sma = new Array(barCount);
    var d_upper_bb = new Array(barCount);
    var d_lower_bb = new Array(barCount);
    var d_roc = new Array(barCount);
    var d_stochastic_k = new Array(barCount);
    var d_stochastic_d = new Array(barCount);
    var d_vol = new Array(barCount);
    var d_vpt = new Array(barCount);
    var d_obv = new Array(barCount);
    var d_kci = new Array(barCount);
    var d_dpv = new Array(barCount);
    var d_apv = new Array(barCount);
    var d_anv = new Array(barCount);

    let annotations = [];

    getRandomData(initialDateStr, fibonacciLevels, lastTimestamp, barData, lineData, d_sma, d_ema, d_rsi, d_macd, d_signal, d_raw_signal, d_histogram, d_bb_sma, d_upper_bb, d_lower_bb, d_roc, d_stochastic_k, d_stochastic_d, d_vol, d_vpt, d_obv, d_kci, d_dpv, d_apv, d_anv);
/*
    for (let i = 0; i < fibonacciLevels.length - 1; i++) {
        let currentFib = fibonacciLevels[i];
        let nextFib = fibonacciLevels[i + 14 < fibonacciLevels.length? i + 14 : i + 1];

        Object.keys(currentFib.levels).forEach(level => {
            annotations.push({
                type: 'line',
                //mode: 'horizontal',
                yScaleID: 'y',
                xScaleID: 'x',
                xMin: currentFib.x,
                yMin: currentFib.levels[level],
                xMax: nextFib.x,
                yMax: nextFib.levels[level],
                borderColor: 'rgba(255, 99, 132, 0.2)',
                borderWidth: 1,
                label: {
                    enabled: false,
                    content: `${level} (${currentFib.levels[level].toFixed(2)})`
                }
            });
        });
    }*/
    
    //console.log("ANNOTATIONS:")
    //console.log(annotations)
    var helperChart = new Chart(ctx2, {
    type: 'candlestick',
    options: {
        scales: {
            y:{
                type:'linear',
                position:'right',
                display:true
            },
            y1: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'left',
                //min: -1,
                //max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y2: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
                //min: -1,
                //max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y3: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
                //min: -1,
                //max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y4: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
        
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y5: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
        
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            }
        },
        plugins: {
            legend:
            {
                position: 'left',
                labels:{
                    boxWidth: 12,
                    padding: 8
                }
            }
        }
    },
    plugins:[predictionPluginHelper/*verticalLinePlugin*/],
    data: {
        datasets: [{
        label: 'ASSET',
        data: barData,
        hidden:true
        },
        {
            label: 'Volume',
            yAxisID: 'y1',
            type: 'line',
            data: d_vol,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:false,
            fill: false
        },
        {
            label: 'RSI',
            type: 'line',
            data: d_rsi,
            borderColor: 'rgba(255, 159, 64, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y4',
            hidden:false, 
            fill: false
        },
        {
            label: 'Raw Signal',
            data: d_raw_signal,
            type: 'line',
            borderColor: 'rgba(255, 0, 10, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },
        {
            label: 'ROC',
            yAxisID: 'y3',
            data: d_roc,
            type: 'line',
            borderColor: 'rgba(0, 250, 10, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:false,
            fill: false
        },
        {
            label: 'Stoch.%K',
            yAxisID: 'y4',
            data: d_stochastic_k,
            type: 'line',
            borderColor: 'rgba(0, 0, 210, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:false,
            fill: false
        },
        {
            label: 'Stoch.%D',
            yAxisID: 'y4',
            data: d_stochastic_d,
            type: 'line',
            borderColor: 'rgba(150, 10, 0, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:false,
            fill: false
        }]
    }
    });



    var chart = new Chart(ctx, {
    type: 'candlestick',
    options: {
        scales: {
            y:{
                type:'linear',
                position:'right',
                display:true
            },
            y1: {
                type: 'linear',
                display: true,
                stacked: true,
                position: 'left',
                min: -1,
                max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y2: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
                min: -1,
                max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y3: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
                min: -1,
                max: 1,
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y4: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
        
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            y5: {
                type: 'linear',
                display: false,
                stacked: true,
                position: 'right',
        
                // grid line settings
                grid: {
                  drawOnChartArea: true, // only want the grid lines for one axis to show up
                },
            },
            x:{
                display:true,
                labels:
                {
                    display:false
                }
            }
        },
        plugins: {
            annotation: {
                annotations: annotations
            },
            legend:
            {
                position: 'left',
                labels:{
                    boxWidth: 12,
                    padding: 8
                }
            },
            customCanvasBackgroundImage: true,
            fibonacciTimeZones:true,            
        }
    },
    plugins: [fibplugin, imgplugin, predictionPluginChart/*, verticalLinePlugin*/],
    data: {
        datasets: [{
        label: 'ASSET',
        data: barData,
        }, {
        label: 'Close price',
        type: 'line',
        data: lineData,
        hidden: true,
        },
        /*{
            label: 'Volume',
            type: 'line',
            data: d_vol,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },*/
        {
            label: 'OBV',
            type: 'line',
            data: d_obv,
            borderColor: 'rgba(255, 191, 0, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y2',
            //hidden:true,
            fill: false
        },
        {
            label: 'DPV',
            type: 'line',
            data: d_dpv,
            borderColor: 'rgba(255, 0, 180, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y3',
            hidden:false,
            fill: false
        },
        {
            label: 'KCI',
            type: 'line',
            data: d_kci,
            borderColor: 'rgba(205, 255, 0, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y1',
            //hidden:true,
            fill: false
        },
        /*{
            label: 'VPT',
            type: 'line',
            data: d_vpt,
            borderColor: 'rgba(103, 0, 255, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },*/
        {
            label: 'EMA',
            type: 'line',
            data: d_ema,
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1,
            pointRadius: 0,
            //hidden:true,
            fill: false
        },
        {
            label: 'RSI',
            type: 'line',
            data: d_rsi,
            borderColor: 'rgba(255, 159, 64, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y4',
            hidden:true,
            fill: false
        },
        /*{
            label: 'MACD',
            type: 'line',
            data: d_macd,
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y5',
            hidden:false,
            fill: false
        },*/
        /*{
            label: 'Signal Line',
            data: d_signal,
            type: 'line',
            borderColor: 'rgba(255, 206, 86, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },*/
        {
            label: 'Raw Signal',
            data: d_raw_signal,
            type: 'line',
            borderColor: 'rgba(255, 0, 10, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },
        {
            label: 'Histogram',
            data: d_histogram,
            type: 'line',
            borderColor: 'rgba(180, 50, 10, 1)',
            borderWidth: 1,
            pointRadius: 0,
            yAxisID: 'y3',
            hidden:true,
            fill: false
        },
        {
            label: 'Upper BB',
            data: d_upper_bb,
            type: 'line',
            borderColor: 'rgba(0, 150, 110, 1)',
            borderWidth: 1,
            pointRadius: 0,
            //hidden:true,
            fill: false
        },
        {
            label: 'Lower BB',
            data: d_lower_bb,
            type: 'line',
            borderColor: 'rgba(0, 150, 110, 1)',
            borderWidth: 1,
            pointRadius: 0,
            //hidden:true,
            fill: false
        },
        {
            label: 'BB SMA',
            data: d_bb_sma,
            type: 'line',
            borderColor: 'rgba(150, 150, 110, 1)',
            borderWidth: 1,
            pointRadius: 0,
            //hidden:true,
            fill: false
        },
        /*{
            label: 'ROC',
            data: d_roc,
            type: 'line',
            borderColor: 'rgba(0, 250, 10, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },
        {
            label: 'Stochastic %K',
            data: d_stochastic_k,
            type: 'line',
            borderColor: 'rgba(0, 0, 210, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        },
        {
            label: 'Stochastic %D',
            data: d_stochastic_d,
            type: 'line',
            borderColor: 'rgba(150, 10, 0, 1)',
            borderWidth: 1,
            pointRadius: 0,
            hidden:true,
            fill: false
        }*/]
    }
    });
    chart.heatmapData = heatmapData

    var macdColors = d_macd.map(point => point.y >= 0 ? 'rgba(0, 255, 0, 0.5)' : 'rgba(255, 0, 0, 0.5)');
    console.log("DMACD")
    console.log(d_macd)
    var chart4 = new Chart(ctx4, {
        type: 'bar',
        data: {
            labels: barData.map(item => item.x),
            datasets: [
                
                {
                    label: '+ Vol.',
                    data: d_apv,
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    borderColor: 'rgba(0, 0, 0, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                },
                {
                    label: '-Vol.',
                    data: d_anv,
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    borderColor: 'rgba(0, 0, 0, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                },
/*
                {
                    label: 'Raw Signal',
                    data: d_raw_signal,
                    type: 'line',
                    borderColor: 'rgba(255, 0, 10, 1)',
                    borderWidth: 1,
                    pointRadius: 0,
                    hidden:false,
                    fill: false,
                    yAxisID: 'y1',
                },*/
                
                {
                    label: 'MACD',
                    data: d_macd,
                    backgroundColor: macdColors,
                    borderColor: macdColors,
                    borderWidth: 1,
                    yAxisID: 'y',
                },
                {
                    label: 'DIF',
                    type: 'line',
                    data: d_macd,
                    borderColor: 'rgba(0, 0, 255, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,    
                    yAxisID: 'y',
                },
                {
                    label: 'DPV',
                    type: 'line',
                    data: d_dpv,
                    borderColor: 'rgba(255, 0, 180, 0.6)',
                    borderWidth: 1,
                    pointRadius: 0,
                    yAxisID: 'y',
                    hidden:false,
                    fill: false
                },
                {
                    label: 'DEA',
                    type: 'line',
                    data: d_signal,
                    borderColor: 'rgba(255, 165, 0, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'y',
                }
            ]
        },
        options: {
            scales: {
                x:  chart.options.scales.x,
                y: {
                    type: 'logarithmic',
                    display: true,
                    stacked: false,
                    position: 'left',
                    //min: -1,
                    //max: 1,
                    // grid line settings
                    grid: {
                      drawOnChartArea: true, // only want the grid lines for one axis to show up
                    },
                },/*
                y1:{
                    type:'logarithmic',
                    position:'right',
                    stacked: true,
                    display:true
                },*/
                
            },
            plugins: {
                legend:
                {
                    position: 'left',
                    labels:{
                        filter: function(label) {
                            if (!label.text.includes('Vol')) return true;
                         },
                        boxWidth: 12,
                        padding: 8
                    }
                }
            },
            layout: {
                padding: {
                    left:28,
                    right:42
                }  // Ensure the layout padding matches
            },
            slopeAnglePlugin:true
        },
        plugins: [slopeAngles, predictionPlugin4/*, verticalLinePlugin*/]
    });

/*
    if(heatmapData)
        {
            const priceArray = heatmapData.liqHeatMap.priceArray;

            // Find min and max values
            let minValue = Number.POSITIVE_INFINITY;
            let maxValue = Number.NEGATIVE_INFINITY;
    
    
            for (let i = 0; i < priceArray.length; i++) {
                    const value = parseFloat(priceArray[i][2]);
                    if (value < minValue) minValue = value;
                    if (value > maxValue) maxValue = value;
            }

            chart.options.scales['y0'].min = minValue;
            chart.options.scales['y0'].max = maxValue;
        }*/

    window.chart=chart
    
    const { c4top, c4left, c4bottom, c4right, c4width, height } = chart.chartArea;
    chart4.chartArea.right = c4right
    chart4.chartArea.left = c4left
    chart4.chartArea.width = c4width

    //console.log(barData)
    //console.log(lineData)
    
    chart.options.responsive = true;
    chart.options.maintainAspectRatio = false;
    
    helperChart.options.responsive = true;
    helperChart.options.maintainAspectRatio = false;

    chart4.options.responsive = true;
    chart4.options.maintainAspectRatio = false;
    
    chart.update();
    helperChart.update();
    chart4.update();

    

    var update = function() {
    var dataset = chart.config.data.datasets[0];
    var dataset2 = helperChart.config.data.datasets[0];
    var dataset3 = chart4.config.data.datasets[0];
    //var dataset = chart.config.data.datasets[0];

    // candlestick vs ohlc
    var type = document.getElementById('type').value;
    chart.config.type = type;
    helperChart.config.type = type;
    chart4.config.type = type;

    // linear vs log
    var scaleType = document.getElementById('scale-type').value;
    chart.config.options.scales.y.type = scaleType;
    helperChart.config.options.scales.y.type = scaleType;
    chart4.config.options.scales.y.type = scaleType;

    // color
    var colorScheme = document.getElementById('color-scheme').value;
    if (colorScheme === 'neon') {
        chart.config.data.datasets[0].backgroundColors = {
        up: '#01ff01',
        down: '#fe0000',
        unchanged: '#999',
        };
        helperChart.config.data.datasets[0].backgroundColors = {
        up: '#01ff01',
        down: '#fe0000',
        unchanged: '#999',
        };
    } else {
        delete chart.config.data.datasets[0].backgroundColors;
        delete helperChart.config.data.datasets[0].backgroundColors;
    }

    // border
    var border = document.getElementById('border').value;
    if (border === 'false') {
        dataset.borderColors = 'rgba(0, 0, 0, 0)';
        dataset2.borderColors = 'rgba(0, 0, 0, 0)';
    } else {
        delete dataset.borderColors;
        delete dataset2.borderColors;
    }

    // mixed charts
    var mixed = document.getElementById('mixed').value;
    if (mixed === 'true') {
        chart.config.data.datasets[1].hidden = false;
        helperChart.config.data.datasets[1].hidden = false;
    } else {
        chart.config.data.datasets[1].hidden = true;
        helperChart.config.data.datasets[1].hidden = true;
    }

    chart.update();
    helperChart.update();
    chart4.update();
    };

    [...document.getElementsByTagName('select')].forEach(element => element.addEventListener('change', update));

    document.getElementById('update').addEventListener('click', update);

    /*document.getElementById('randomizeData').addEventListener('click', function() {
    getRandomData(initialDateStr, barData);
    update();
    });*/


    var dataset = chart.config.data.datasets[0];
    var dataset2 = helperChart.config.data.datasets[0];

    // candlestick vs ohlc
    var type = document.getElementById('type').value;
    chart.config.type = type;
    helperChart.config.type = type;
  
    // linear vs log
    var scaleType = document.getElementById('scale-type').value;
    chart.config.options.scales.y.type = scaleType;
    helperChart.config.options.scales.y.type = scaleType;
    chart4.config.options.scales.y.type = scaleType;
  
    // color
    var colorScheme = document.getElementById('color-scheme').value;
    if (colorScheme === 'neon') {
      chart.config.data.datasets[0].backgroundColors = {
        up: '#01ff01',
        down: '#fe0000',
        unchanged: '#999',
      };
      helperChart.config.data.datasets[0].backgroundColors = {
        up: '#01ff01',
        down: '#fe0000',
        unchanged: '#999',
      };
    } else {
      delete chart.config.data.datasets[0].backgroundColors;
      delete helperChart.config.data.datasets[0].backgroundColors;
    }
  
    // border
    var border = document.getElementById('border').value;
    if (border === 'false') {
      dataset.borderColors = 'rgba(0, 0, 0, 0)';
      dataset2.borderColors = 'rgba(0, 0, 0, 0)';
    } else {
      delete dataset.borderColors;
      delete dataset2.borderColors;
    }
  
    // mixed charts
    var mixed = document.getElementById('mixed').value;
    if (mixed === 'true') {
      chart.config.data.datasets[1].hidden = false;
      helperChart.config.data.datasets[1].hidden = false;
    } else {
      chart.config.data.datasets[1].hidden = true;
      helperChart.config.data.datasets[1].hidden = true;
    }
  
    canvas.style.width = width;
    chart.update();
    chart4.update();
    helperChart.update();

    const verticalLine = document.getElementById('vertical-line');

        
    // Add event listener for mouse movement over the canvas
    document.getElementById('chart').addEventListener('mousemove', (event) => {
        const rect = event.target.getBoundingClientRect();
        cursorX = event.clientX - rect.left;
        
        verticalLine.style.left = `${event.clientX}px`;
        verticalLine.style.top = '0px';
        verticalLine.style.display = 'block';
        //chart.update('none');
        //helperChart.update('none');
        //chart4.update('none');
    });

    // Add event listener for mouse leave to clear the line
    document.getElementById('chart').addEventListener('mouseleave', () => {
        cursorX = null;
        verticalLine.style.display = 'none';
        //chart.update('none');
        //helperChart.update('none');
        //chart4.update('none');
    });

    // Add event listener for mouse movement over the canvas
    document.getElementById('chart2').addEventListener('mousemove', (event) => {
        const rect = event.target.getBoundingClientRect();
        cursorX = event.clientX - rect.left;
        
        verticalLine.style.left = `${event.clientX}px`;
        verticalLine.style.top = '0px';
        verticalLine.style.display = 'block';
        //chart.update('none');
        //helperChart.update('none');
        //chart4.update('none');
    });

    // Add event listener for mouse leave to clear the line
    document.getElementById('chart2').addEventListener('mouseleave', () => {
        cursorX = null;
        verticalLine.style.display = 'none';
        //chart.update('none');
        //helperChart.update('none');
        //chart4.update('none');
    });

    // Add event listener for mouse movement over the canvas
    document.getElementById('chart4').addEventListener('mousemove', (event) => {
        const rect = event.target.getBoundingClientRect();
        cursorX = event.clientX - rect.left;
        
        verticalLine.style.left = `${event.clientX}px`;
        verticalLine.style.top = '0px';
        verticalLine.style.display = 'block';
        //chart.update('none');
        //helperChart.update('none');
        //chart4.update('none');
    });

    // Add event listener for mouse leave to clear the line
    document.getElementById('chart4').addEventListener('mouseleave', () => {
        cursorX = null;
        verticalLine.style.display = 'none';
        //chart4.update('none');
    });

    return {chart:chart, helper:helperChart, vhelper: chart4}
}

eel.expose(generateChart);