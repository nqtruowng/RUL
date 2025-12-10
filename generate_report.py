"""
Generate HTML Report from report.csv
Run this after inference to create report.html with embedded data
"""

import pandas as pd
import os

def generate_html_report(csv_path='report.csv', output_path='report.html'):
    """Generate HTML report with embedded CSV data"""
    
    print("Reading CSV data...")
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to JavaScript-friendly format
    js_data = []
    for _, row in df.iterrows():
        js_data.append({
            'engine_id': int(row['engine_id']),
            'predicted_RUL': round(float(row['predicted_RUL']), 2),
            'true_RUL': int(row['true_RUL']),
            'imminent_failure_flag': row['imminent_failure_flag'] == True or str(row['imminent_failure_flag']).lower() == 'true',
            'top_sensors': str(row['top_sensors']),
            'top_subsystems': str(row['top_subsystems'])
        })
    
    # Generate JavaScript array string
    js_array = "[\n"
    for i, item in enumerate(js_data):
        js_array += f"""            {{
                engine_id: {item['engine_id']},
                predicted_RUL: {item['predicted_RUL']},
                true_RUL: {item['true_RUL']},
                imminent_failure_flag: {'true' if item['imminent_failure_flag'] else 'false'},
                top_sensors: `{item['top_sensors']}`,
                top_subsystems: `{item['top_subsystems']}`
            }}"""
        if i < len(js_data) - 1:
            js_array += ",\n"
        else:
            js_array += "\n"
    js_array += "        ]"
    
    html_content = f'''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Engine RUL Prediction Report</title>
    <style>
        :root {{
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --accent-cyan: #00d9ff;
            --accent-magenta: #ff006e;
            --accent-yellow: #ffbe0b;
            --accent-green: #06d6a0;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --border-color: #2a2a4a;
            --danger: #ef476f;
            --warning: #ffd166;
            --success: #06d6a0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
            padding: 2rem 3rem;
            border-bottom: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta), var(--accent-yellow));
        }}

        .header h1 {{
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}

        .header h1 span {{
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .summary-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .summary-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 217, 255, 0.15);
        }}

        .summary-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }}

        .summary-card.cyan::before {{ background: var(--accent-cyan); }}
        .summary-card.magenta::before {{ background: var(--accent-magenta); }}
        .summary-card.yellow::before {{ background: var(--accent-yellow); }}
        .summary-card.green::before {{ background: var(--accent-green); }}
        .summary-card.danger::before {{ background: var(--danger); }}

        .summary-card .label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}

        .summary-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1;
        }}

        .summary-card.cyan .value {{ color: var(--accent-cyan); }}
        .summary-card.magenta .value {{ color: var(--accent-magenta); }}
        .summary-card.yellow .value {{ color: var(--accent-yellow); }}
        .summary-card.green .value {{ color: var(--accent-green); }}
        .summary-card.danger .value {{ color: var(--danger); }}

        .summary-card .subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}

        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .section-title::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: var(--accent-cyan);
            border-radius: 2px;
        }}

        .alert-banner {{
            background: linear-gradient(135deg, rgba(239, 71, 111, 0.15) 0%, rgba(255, 0, 110, 0.1) 100%);
            border: 1px solid var(--danger);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .alert-icon {{
            width: 48px;
            height: 48px;
            background: var(--danger);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
        }}

        .alert-content h3 {{
            color: var(--danger);
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }}

        .alert-content p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .table-container {{
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            margin-bottom: 2rem;
        }}

        .table-scroll {{
            max-height: 500px;
            overflow-y: auto;
        }}

        .table-header {{
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .table-header h3 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .filter-tabs {{
            display: flex;
            gap: 0.5rem;
        }}

        .filter-tab {{
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}

        .filter-tab:hover {{
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}

        .filter-tab.active {{
            background: var(--accent-cyan);
            border-color: var(--accent-cyan);
            color: var(--bg-primary);
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .data-table th {{
            background: var(--bg-secondary);
            padding: 1rem 1.25rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        .data-table td {{
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.9rem;
        }}

        .data-table tbody tr {{
            transition: background 0.2s ease;
        }}

        .data-table tbody tr:hover {{
            background: rgba(0, 217, 255, 0.05);
        }}

        .data-table tbody tr.critical {{
            background: rgba(239, 71, 111, 0.1);
        }}

        .data-table tbody tr.critical:hover {{
            background: rgba(239, 71, 111, 0.15);
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .status-badge.critical {{
            background: rgba(239, 71, 111, 0.2);
            color: var(--danger);
        }}

        .status-badge.warning {{
            background: rgba(255, 209, 102, 0.2);
            color: var(--warning);
        }}

        .status-badge.normal {{
            background: rgba(6, 214, 160, 0.2);
            color: var(--success);
        }}

        .status-badge::before {{
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: currentColor;
        }}

        .rul-bar {{
            width: 100%;
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.25rem;
        }}

        .rul-bar-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .rul-bar-fill.critical {{ background: var(--danger); }}
        .rul-bar-fill.warning {{ background: var(--warning); }}
        .rul-bar-fill.normal {{ background: var(--success); }}

        .sensor-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}

        .sensor-tag {{
            padding: 0.25rem 0.6rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--accent-cyan);
            border: 1px solid var(--border-color);
        }}

        .subsystem-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .subsystem-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
        }}

        .subsystem-card h4 {{
            font-size: 1rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }}

        .subsystem-item {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }}

        .subsystem-name {{
            flex: 1;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .subsystem-bar {{
            width: 120px;
            height: 6px;
            background: var(--bg-secondary);
            border-radius: 3px;
            overflow: hidden;
        }}

        .subsystem-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
            border-radius: 3px;
        }}

        .subsystem-value {{
            width: 50px;
            text-align: right;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }}

        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--bg-secondary);
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--accent-cyan);
        }}

        @media (max-width: 768px) {{
            .header {{
                padding: 1.5rem;
            }}
            
            .container {{
                padding: 1rem;
            }}

            .summary-grid {{
                grid-template-columns: 1fr 1fr;
            }}

            .table-container {{
                overflow-x: auto;
            }}

            .data-table {{
                min-width: 800px;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <h1><span>Engine RUL</span> Prediction Report</h1>
    </header>

    <main class="container">
        <section class="summary-grid" id="summary-cards"></section>

        <div class="alert-banner" id="alert-banner" style="display: none;">
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <h3>Critical Engines Detected</h3>
                <p id="alert-text">Several engines require immediate maintenance attention.</p>
            </div>
        </div>

        <h2 class="section-title">Subsystem Impact Analysis</h2>
        <section class="subsystem-section" id="subsystem-analysis"></section>

        <div class="table-container">
            <div class="table-header">
                <h3>Engine Status Details</h3>
                <div class="filter-tabs">
                    <button class="filter-tab active" onclick="filterTable('all')">All Engines</button>
                    <button class="filter-tab" onclick="filterTable('critical')">Critical Only</button>
                    <button class="filter-tab" onclick="filterTable('warning')">Warning</button>
                    <button class="filter-tab" onclick="filterTable('normal')">Normal</button>
                </div>
            </div>
            <div class="table-scroll">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Engine ID</th>
                            <th>Predicted RUL</th>
                            <th>True RUL</th>
                            <th>Error</th>
                            <th>Status</th>
                            <th>Top Contributing Sensors</th>
                        </tr>
                    </thead>
                    <tbody id="engine-table-body"></tbody>
                </table>
            </div>
        </div>
    </main>

    <script>
        // Embedded data from report.csv
        const engines = {js_array};

        // Calculate statistics
        function calculateStats(data) {{
            const total = data.length;
            const critical = data.filter(e => e.predicted_RUL <= 30).length;
            const warning = data.filter(e => e.predicted_RUL > 30 && e.predicted_RUL <= 60).length;
            const avgRUL = data.reduce((sum, e) => sum + e.predicted_RUL, 0) / total;
            const avgError = data.reduce((sum, e) => sum + Math.abs(e.predicted_RUL - e.true_RUL), 0) / total;
            return {{ total, critical, warning, avgRUL, avgError }};
        }}

        const stats = calculateStats(engines);

        function renderSummaryCards() {{
            document.getElementById('summary-cards').innerHTML = `
                <div class="summary-card cyan">
                    <div class="label">Total Engines</div>
                    <div class="value">${{stats.total}}</div>
                    <div class="subtext">Analyzed with XAI</div>
                </div>
                <div class="summary-card danger">
                    <div class="label">Critical (RUL ≤ 30)</div>
                    <div class="value">${{stats.critical}}</div>
                    <div class="subtext">Immediate attention needed</div>
                </div>
                <div class="summary-card yellow">
                    <div class="label">Warning (30 < RUL ≤ 60)</div>
                    <div class="value">${{stats.warning}}</div>
                    <div class="subtext">Schedule maintenance</div>
                </div>
                <div class="summary-card green">
                    <div class="label">Average Predicted RUL</div>
                    <div class="value">${{stats.avgRUL.toFixed(1)}}</div>
                    <div class="subtext">cycles remaining</div>
                </div>
                <div class="summary-card magenta">
                    <div class="label">Mean Absolute Error</div>
                    <div class="value">${{stats.avgError.toFixed(1)}}</div>
                    <div class="subtext">cycles deviation</div>
                </div>
            `;
        }}

        function renderAlertBanner() {{
            if (stats.critical > 0) {{
                document.getElementById('alert-banner').style.display = 'flex';
                document.getElementById('alert-text').textContent = 
                    `${{stats.critical}} engine(s) with RUL ≤ 30 cycles require immediate maintenance. ` +
                    `IDs: ${{engines.filter(e => e.predicted_RUL <= 30).map(e => e.engine_id).join(', ')}}`;
            }}
        }}

        function renderSubsystemAnalysis() {{
            const subsystems = {{}};
            engines.forEach(e => {{
                const match = e.top_subsystems.match(/\\('([^']+)', ([\\d.]+)\\)/g);
                if (match) {{
                    match.forEach(m => {{
                        const [, name, score] = m.match(/\\('([^']+)', ([\\d.]+)\\)/);
                        subsystems[name] = (subsystems[name] || 0) + parseFloat(score);
                    }});
                }}
            }});

            const sorted = Object.entries(subsystems).sort((a, b) => b[1] - a[1]).slice(0, 8);
            const maxScore = sorted[0][1];

            document.getElementById('subsystem-analysis').innerHTML = `
                <div class="subsystem-card">
                    <h4>Top Affected Subsystems (Aggregated)</h4>
                    ${{sorted.map(([name, score]) => `
                        <div class="subsystem-item">
                            <span class="subsystem-name">${{name}}</span>
                            <div class="subsystem-bar">
                                <div class="subsystem-bar-fill" style="width: ${{(score / maxScore * 100).toFixed(1)}}%"></div>
                            </div>
                            <span class="subsystem-value">${{(score / engines.length * 100).toFixed(1)}}%</span>
                        </div>
                    `).join('')}}
                </div>
                <div class="subsystem-card">
                    <h4>Critical Engines Overview</h4>
                    ${{engines.filter(e => e.predicted_RUL <= 30).slice(0, 6).map(e => `
                        <div class="subsystem-item">
                            <span class="subsystem-name">Engine #${{e.engine_id}}</span>
                            <div class="subsystem-bar">
                                <div class="subsystem-bar-fill" style="width: ${{(e.predicted_RUL / 30 * 100).toFixed(1)}}%; background: var(--danger);"></div>
                            </div>
                            <span class="subsystem-value" style="color: var(--danger);">${{e.predicted_RUL.toFixed(1)}}</span>
                        </div>
                    `).join('')}}
                </div>
            `;
        }}

        function getStatus(rul) {{
            if (rul <= 30) return 'critical';
            if (rul <= 60) return 'warning';
            return 'normal';
        }}

        function renderTable(filter = 'all') {{
            let filteredData = engines;
            if (filter === 'critical') filteredData = engines.filter(e => e.predicted_RUL <= 30);
            else if (filter === 'warning') filteredData = engines.filter(e => e.predicted_RUL > 30 && e.predicted_RUL <= 60);
            else if (filter === 'normal') filteredData = engines.filter(e => e.predicted_RUL > 60);

            document.getElementById('engine-table-body').innerHTML = filteredData.map(e => {{
                const status = getStatus(e.predicted_RUL);
                const error = (e.predicted_RUL - e.true_RUL).toFixed(1);
                const errorStyle = parseFloat(error) > 0 ? 'color: var(--danger)' : 'color: var(--success)';
                const sensors = e.top_sensors.match(/sensor_\\d+/g) || [];

                return `
                    <tr class="${{status === 'critical' ? 'critical' : ''}}">
                        <td><strong>#${{e.engine_id}}</strong></td>
                        <td>
                            <div>${{e.predicted_RUL.toFixed(1)}} cycles</div>
                            <div class="rul-bar">
                                <div class="rul-bar-fill ${{status}}" style="width: ${{Math.min(e.predicted_RUL / 125 * 100, 100)}}%"></div>
                            </div>
                        </td>
                        <td>${{e.true_RUL}} cycles</td>
                        <td style="${{errorStyle}}">${{error > 0 ? '+' : ''}}${{error}}</td>
                        <td><span class="status-badge ${{status}}">${{status}}</span></td>
                        <td>
                            <div class="sensor-tags">
                                ${{sensors.slice(0, 3).map(s => `<span class="sensor-tag">${{s}}</span>`).join('')}}
                            </div>
                        </td>
                    </tr>
                `;
            }}).join('');
        }}

        function filterTable(filter) {{
            document.querySelectorAll('.filter-tab').forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');
            renderTable(filter);
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            renderSummaryCards();
            renderAlertBanner();
            renderSubsystemAnalysis();
            renderTable();
        }});
    </script>
</body>
</html>
'''
    
    print(f"Writing HTML report to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[OK] Report generated: {output_path}")
    print(f"[OK] Total engines: {len(df)}")
    print(f"[OK] Critical engines: {len(df[df['predicted_RUL'] <= 30])}")
    print(f"\nOpen {output_path} directly in browser (no server needed)!")

if __name__ == "__main__":
    generate_html_report()

