export const tonalityButtons = [
	{
		id: 0,
		isCount: true,
		title: 'Негативные упоминания',
	},
	{
		id: 1,
		isCount: true,
		title: 'Позитивные упоминания',
	},
	{
		id: 2,
		isCount: false,
		title: 'Тональность авторов',
	},
];

export const informationButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Граф. распространения информации',
	},
	{
		id: 1,
		isCount: false,
		title: 'Динамика распространения',
	},
	{
		id: 2,
		isCount: false,
		title: 'Динамика по авторам',
	},
];

export const mediaButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Рейтинг тональности в СМИ',
	},
	{
		id: 1,
		isCount: false,
		title: 'Динамика в СМИ',
	},
];

export const voiceButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Источники',
	},
	{
		id: 1,
		isCount: false,
		title: 'Тип упоминаний',
	},
];

export const competitiveButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Динамика сообщений',
	},
	{
		id: 1,
		isCount: false,
		title: 'Сравнение в СМИ и Соцмедиа',
	},
	{
		id: 2,
		isCount: false,
		title: 'Рейтинг СМИ и Соцмедиа в динамике',
	},
];

export const dataSetButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Файлы данных',
	},
	{
		id: 1,
		isCount: false,
		title: 'Файлы кластеризации авторов',
		badge: 'В разработке',
		disabled: true,
	},
	{
		id: 2,
		isCount: false,
		title: 'Статус ИИ-расчета данных',
	},
	{
		id: 3,
		isCount: false,
		title: 'Мосинформ.Рейтинг',
	},
	{
		id: 4,
		isCount: false,
		title: 'Очередь ML',
	},
];

export const analysisOfThemesButtons = [
	{
		id: 0,
		isCount: false,
		title: 'Кластеризация на тематики',
	},
	{
		id: 1,
		isCount: false,
		title: 'Выявленные темы',
	},
	{
		id: 2,
		isCount: false,
		title: 'Анализ тематик',
	},
];

// Styles object для использования в styled-components или CSS-in-JS
export const buttonStyles = {
	disabled: {
		opacity: 0.5,
		cursor: 'not-allowed',
		position: 'relative',
	},
	badge: {
		fontSize: '9px',
		background: '#f39c12',
		color: 'white',
		padding: '2px 6px',
		borderRadius: '10px',
		marginLeft: '8px',
		fontWeight: 500,
	},
};